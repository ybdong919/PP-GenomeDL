# -*- coding: utf-8 -*-

##############################################
# FEDERATED LEARNING + MULTIPARTY CKKS FHE
# VERSION 16 + SIMD OPTIMIZATION (FIXED)
##############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import copy
import logging
import time
import argparse
import random
from sklearn.metrics import f1_score
from math import sqrt, exp
from scipy.special import erf
import gc

##############################################
# FHE CKKS Multiparty
##############################################
try:
    from openfhe import *
    OPENFHE_AVAILABLE = True
except ImportError:
    print("[WARNING] OpenFHE not available. Install with: pip install openfhe-python")
    OPENFHE_AVAILABLE = False

root = '.'

########################################################
# BASE UTILITY CODE
########################################################

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0
    def update(self, val):
        self.total += val
        self.steps += 1
    def __call__(self):
        if self.steps == 0:
            return 0
        return self.total / float(self.steps)

def log_creater(output_dir, expname):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(expname)
    final_log_file = os.path.join(output_dir, log_name)
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    log.handlers = []
    
    file = logging.FileHandler(final_log_file, 'w')
    file.setLevel(logging.DEBUG)
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][line: %(lineno)d] ==> [INFO] %(message)s')
    file.setFormatter(formatter)
    stream.setFormatter(formatter)
    log.addHandler(file)
    log.addHandler(stream)
    log.info('creating {}'.format(final_log_file))
    return log


########################################################
# MODEL (Original v16 architecture)
########################################################
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden3 = nn.Linear(hidden_size // 2, 256)
        self.hidden4 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


########################################################
# DATASET
########################################################

# Global variable for cancer types
cancerTypes = []

def cancerType(type):
    global cancerTypes
    idx = cancerTypes.index(type)
    return idx

class Clinical_Data(Dataset):
    def __init__(self, index):
        data_path = os.path.join(root, "data/{}.npy".format(index))
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print("[INFO] Loading {}".format(data_path))
        self.data = np.load(data_path, allow_pickle=True).item()
        self.samples = self.data['samples']
        self.feature_dic = self.data['features']
        print("[INFO] {} Data has {} samples".format(index, len(self.samples)))

    def __getitem__(self, idx):
        values_list = list(self.samples.values())
        data_dic = copy.deepcopy(values_list[idx])
        genes = data_dic['genes']
        label = cancerType(data_dic['cancertype'])
        del data_dic['genes']
        return torch.Tensor(genes), label

    def __len__(self):
        return len(self.samples)

    def __alltypes__(self):
        out = []
        values_list = list(self.samples.values())
        for idx in range(len(self.samples)):
            out.append(values_list[idx]['cancertype'])
        return out


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y


def noniid_partition(data_size, num_clients, alpha=0.5):
    min_size_per_client = 10
    proportions = np.random.dirichlet([alpha] * num_clients)
    proportions = proportions / proportions.sum()
    min_size = np.ones(num_clients) * min_size_per_client
    total_min_size = min_size.sum()
    
    if total_min_size > data_size:
        raise ValueError(f"Cannot partition {data_size} samples into {num_clients} clients with minimum {min_size_per_client} each")
    
    available_data = data_size - total_min_size
    additional_samples = (proportions * available_data).astype(int)
    total_allocated = min_size + additional_samples
    remainder = int(data_size - total_allocated.sum())
    
    if remainder > 0:
        total_allocated[:remainder] += 1
    
    return total_allocated.astype(int)


def get_dataset_HAV(args):
    global cancerTypes
    
    # Load datasets
    train_dataset = Clinical_Data(args.train_data)
    test_dataset = Clinical_Data(args.test_data)
    
    # Get all cancer types
    cancerTypes = list(set(train_dataset.__alltypes__()))
    cancerTypes.sort()
    
    train_loaders = []
    dict_users = {}
    
    num_samples_per_client = noniid_partition(len(train_dataset), args.client, alpha=0.5)
    
    all_idxs = set(np.arange(len(train_dataset)))
    client_idxs = []
    start_idx = 0
    for i in range(args.client):
        num_samples = num_samples_per_client[i]
        idxs = list(range(start_idx, start_idx + num_samples))
        dict_users[i] = idxs
        client_idxs.append(idxs)
        start_idx += num_samples
    
    for i in range(args.client):
        train_loader = DataLoader(DatasetSplit(train_dataset, dict_users[i]),
                                   batch_size=args.batch_size, shuffle=True)
        train_loaders.append(train_loader)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loaders, test_loader


def Testing(model, testLoader, criterion, device, logger):
    model.to(device)
    model.eval()
    
    test_loss = RunningAverage()
    test_acc = RunningAverage()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for X, y in tqdm(testLoader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            
            test_loss.update(loss.item())
            pred = output.argmax(dim=1)
            acc = (pred == y).float().mean()
            test_acc.update(acc.item())
            
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    
    logger.info(f"TEST: Loss={test_loss():.4f} ACC={test_acc():.4f} Macro F1={macro_f1:.4f} Micro F1={micro_f1:.4f}")
    
    return test_loss(), test_acc()


########################################################
# SIMD FHE FUNCTIONS (FIXED)
########################################################

def init_fhe_simd(logger):
    """Initialize SIMD-optimized CKKS parameters"""
    try:
        parameters = CCParamsCKKSRNS()
        
        # SIMD-optimized parameters
        ring_dim = 65536
        mult_depth = 25
        scale_mod_size = 45
        first_mod_size = 60
        batch_size = 16384
        
        parameters.SetMultiplicativeDepth(mult_depth)
        parameters.SetScalingModSize(scale_mod_size)
        parameters.SetFirstModSize(first_mod_size)
        parameters.SetRingDim(ring_dim)
        parameters.SetBatchSize(batch_size)
        parameters.SetSecurityLevel(HEStd_NotSet)
        parameters.SetKeySwitchTechnique(HYBRID)
        
        logger.info("="*60)
        logger.info("SIMD-OPTIMIZED CKKS CONFIGURATION:")
        logger.info(f"  Ring Dimension: {ring_dim}")
        logger.info(f"  Multiplicative Depth: {mult_depth}")
        logger.info(f"  Scaling Mod Size: {scale_mod_size}")
        logger.info(f"  First Mod Size: {first_mod_size}")
        logger.info(f"  Batch Size (SIMD): {batch_size}")
        logger.info(f"  Optimization: 16x more values per ciphertext vs v16")
        logger.info("="*60)
        
        cc = GenCryptoContext(parameters)
        cc.Enable(PKE)
        cc.Enable(KEYSWITCH)
        cc.Enable(LEVELEDSHE)
        cc.Enable(MULTIPARTY)
        
        return cc, batch_size
        
    except Exception as e:
        logger.error(f"FHE initialization failed: {str(e)}")
        return None, None


def encrypt_gradients_simd(cc, public_key, gradients, chunk_size, logger, 
                           scale_factor=100.0, clip_value=1.0):
    """Encrypt gradients using SIMD packing"""
    encrypted_chunks = []
    
    # Flatten all gradients
    flat_grads = []
    for grad in gradients:
        flat_grads.extend(grad.cpu().numpy().flatten().tolist())
    
    # Clip and scale
    flat_grads = np.clip(flat_grads, -clip_value, clip_value)
    scaled_grads = [g * scale_factor for g in flat_grads]
    
    # Pack into SIMD chunks
    num_chunks = (len(scaled_grads) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(scaled_grads))
        chunk = scaled_grads[start_idx:end_idx]
        
        # Pad if necessary
        if len(chunk) < chunk_size:
            chunk.extend([0.0] * (chunk_size - len(chunk)))
        
        # Encrypt SIMD chunk
        plaintext = cc.MakeCKKSPackedPlaintext(chunk)
        ciphertext = cc.Encrypt(public_key, plaintext)
        encrypted_chunks.append(ciphertext)
    
    return encrypted_chunks, num_chunks


def aggregate_encrypted_gradients_simd(cc, enc_client1, enc_client2, logger):
    """Aggregate encrypted gradients using SIMD operations"""
    aggregated = []
    
    for c1, c2 in zip(enc_client1, enc_client2):
        # Homomorphic addition
        agg = cc.EvalAdd(c1, c2)
        
        # Average (divide by 2)
        agg = cc.EvalMult(agg, 0.5)
        
        aggregated.append(agg)
    
    return aggregated


def decrypt_gradients_simd(cc, secret_keys, encrypted_chunks, chunk_size, logger, 
                           scale_factor=100.0):
    """Decrypt SIMD chunks using multiparty protocol"""
    decrypted_chunks = []
    
    for cipher in encrypted_chunks:
        # Multiparty decryption
        partial_plain1 = cc.MultipartyDecryptLead([cipher], secret_keys[0])
        partial_plain2 = cc.MultipartyDecryptMain([cipher], secret_keys[1])
        
        # Fusion
        partial_plains = [partial_plain1[0], partial_plain2[0]]
        plaintext = cc.MultipartyDecryptFusion(partial_plains)
        
        # Extract and unscale
        plaintext.SetLength(chunk_size)
        values = plaintext.GetRealPackedValue()
        unscaled = [v / scale_factor for v in values]
        
        decrypted_chunks.append(unscaled)
    
    return decrypted_chunks


def reconstruct_gradients_simd(decrypted_chunks, original_shapes):
    """Reconstruct gradients from SIMD chunks"""
    # Flatten all chunks
    flat_values = []
    for chunk in decrypted_chunks:
        flat_values.extend(chunk)
    
    # Reconstruct tensors
    reconstructed = []
    offset = 0
    
    for shape in original_shapes:
        num_elements = np.prod(shape)
        values = flat_values[offset:offset + num_elements]
        tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        reconstructed.append(tensor)
        offset += num_elements
    
    return reconstructed


########################################################
# TRAINING (SIMPLIFIED - SIMD ONLY)
########################################################

def Train(logger, trainLoaders, testLoader, serverModel, criterions, device, 
          num_epochs=10, use_fhe=True, num_clients=2):
    """
    SIMD-only training without adaptive scaling
    """
    
    # Initialize SIMD FHE
    cc = None
    public_key = None
    secret_keys = None
    
    if use_fhe and OPENFHE_AVAILABLE:
        try:
            logger.info("Initializing SIMD-OPTIMIZED FHE...")
            cc, batch_size = init_fhe_simd(logger)
            
            if cc is not None:
                # FIXED: Multiparty key generation using private keys
                key_pair1 = cc.KeyGen()
                key_pair2 = cc.MultipartyKeyGen(key_pair1.publicKey)
                
                secret_keys = [key_pair1.secretKey, key_pair2.secretKey]
                
                # FIXED: Joint public key from private keys
                private_key_list = [key_pair1.secretKey, key_pair2.secretKey]
                joint_key_pair = cc.MultipartyKeyGen(private_key_list)
                public_key = joint_key_pair.publicKey
                
                logger.info("✓ SIMD FHE initialized successfully!")
            
        except Exception as e:
            logger.error(f"SIMD FHE initialization failed: {str(e)}")
            logger.info("Continuing WITHOUT FHE encryption...")
            use_fhe = False
    else:
        logger.info("Running WITHOUT FHE encryption")
    
    # SIMD FHE parameters (FIXED - no adaptive scaling)
    chunk_size = 16384
    scale_factor = 100.0
    clip_value = 1.0
    
    logger.info(f"SIMD Configuration: chunk_size={chunk_size} (16x larger than v16)")
    logger.info(f"Expected ciphertexts: ~{10719013 // chunk_size} (vs v16's 2622)")
    
    serverModel.to(device)
    serverModel.train()
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        loss_avg = RunningAverage()
        labels = []
        preds = []
        
        initial_server_model = copy.deepcopy(serverModel)
        
        # Train clients locally
        client_models = []
        client_updates = []
        
        for client_idx in range(num_clients):
            logger.info(f"Training Client {client_idx+1}...")
            client_model = copy.deepcopy(initial_server_model)
            client_model.to(device)
            client_model.train()
            
            optimizer = torch.optim.Adam(client_model.parameters(), lr=1e-4)
            
            for batch_idx, (X, y) in enumerate(trainLoaders[client_idx]):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = client_model(X)
                loss = criterions(output, y)
                loss.backward()
                optimizer.step()
                
                loss_avg.update(loss.item())
                labels.extend(y.cpu().numpy())
                preds.extend(output.detach().cpu().numpy())
            
            client_models.append(client_model)
            
            # Compute updates
            updates = []
            for server_param, client_param in zip(initial_server_model.parameters(), 
                                                   client_model.parameters()):
                update = client_param.data - server_param.data
                updates.append(update)
            client_updates.append(updates)
            
            logger.info(f"  Client {client_idx+1} completed")
        
        # FHE aggregation with SIMD
        if use_fhe and cc is not None:
            logger.info("Performing SIMD FHE aggregation...")
            logger.info(f"Processing {len(client_updates[0])} layers...")
            
            try:
                # SIMD Encryption
                logger.info("  SIMD encrypting Client 1 gradients...")
                enc_client1, count1 = encrypt_gradients_simd(
                    cc, public_key, client_updates[0], chunk_size, logger,
                    scale_factor=scale_factor, clip_value=clip_value
                )
                logger.info(f"    Client 1: {count1} SIMD chunks (v16 would use {count1*16} chunks)")
                
                logger.info("  SIMD encrypting Client 2 gradients...")
                enc_client2, count2 = encrypt_gradients_simd(
                    cc, public_key, client_updates[1], chunk_size, logger,
                    scale_factor=scale_factor, clip_value=clip_value
                )
                logger.info(f"    Client 2: {count2} SIMD chunks")
                
                # SIMD Aggregation
                logger.info(f"  SIMD aggregating {count1} chunks...")
                aggregated_enc = aggregate_encrypted_gradients_simd(
                    cc, enc_client1, enc_client2, logger
                )
                
                # SIMD Decryption
                logger.info(f"  SIMD decrypting {len(aggregated_enc)} chunks...")
                decrypted_chunks = decrypt_gradients_simd(
                    cc, secret_keys, aggregated_enc, chunk_size, logger,
                    scale_factor=scale_factor
                )
                
                # Reconstruct
                logger.info("  Reconstructing gradients...")
                original_shapes = [p.shape for p in client_updates[0]]
                aggregated_updates = reconstruct_gradients_simd(decrypted_chunks, original_shapes)
                
                # Validation
                total_gradients = sum(np.prod(shape) for shape in original_shapes)
                reconstructed_count = sum(grad.numel() for grad in aggregated_updates)
                
                if reconstructed_count != total_gradients:
                    logger.error(f"Gradient count mismatch! Expected {total_gradients}, got {reconstructed_count}")
                
                # Check for NaN/Inf
                nan_count = sum(torch.isnan(grad).sum().item() for grad in aggregated_updates)
                inf_count = sum(torch.isinf(grad).sum().item() for grad in aggregated_updates)
                
                if nan_count > 0 or inf_count > 0:
                    logger.warning(f"  Found {nan_count} NaN and {inf_count} Inf values after SIMD reconstruction")
                else:
                    logger.info(f"    ✓ All {total_gradients} gradient values are valid!")
                
                # Compute gradient statistics
                grad_norms = [grad.norm().item() for grad in aggregated_updates]
                logger.info(f"    Gradient norms: avg={np.mean(grad_norms):.6f}, max={np.max(grad_norms):.6f}")
                
            except Exception as e:
                logger.error(f"SIMD FHE aggregation failed: {str(e)}")
                logger.info("Falling back to plaintext aggregation...")
                aggregated_updates = []
                for i in range(len(client_updates[0])):
                    avg_update = (client_updates[0][i] + client_updates[1][i]) / 2.0
                    aggregated_updates.append(avg_update)
        else:
            # Plaintext aggregation
            aggregated_updates = []
            for i in range(len(client_updates[0])):
                avg_update = (client_updates[0][i] + client_updates[1][i]) / 2.0
                aggregated_updates.append(avg_update)
        
        # Update server model
        for server_param, update in zip(serverModel.parameters(), aggregated_updates):
            server_param.data += update
        
        # Compute weight norms
        weight_norms = [p.norm().item() for p in serverModel.parameters()]
        logger.info(f"    Weight norms: avg={np.mean(weight_norms):.4f}, max={np.max(weight_norms):.4f}")
        
        # Compute prediction distribution
        unique_preds = len(np.unique(np.array(preds).argmax(axis=1)))
        logger.info(f"    Prediction distribution: {unique_preds} unique classes")
        
        # Metrics
        acc = (torch.tensor(preds).argmax(dim=1) == torch.tensor(labels).squeeze()).sum() / len(labels)
        acc = acc.item()
        macro_f1 = f1_score(labels, torch.tensor(preds).argmax(dim=1), average='macro')
        
        logger.info(f"Epoch {epoch+1}: Loss={loss_avg():.4f} ACC={acc:.4f} F1={macro_f1:.4f}")
        
        # Test
        test_loss, test_acc = Testing(serverModel, testLoader, criterions, device, logger)
        
        gc.collect()
    
    return serverModel


########################################################
# MAIN
########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--client', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--expname', type=str, default='v16_simd_fixed')
    parser.add_argument('--train_data', type=str, default='train_data_norm_log10')
    parser.add_argument('--test_data', type=str, default='test_data_norm_log10')
    parser.add_argument('--no_fhe', action='store_true', help='Disable FHE encryption')
    args = parser.parse_args()
    
    logger = log_creater(output_dir='./log', expname=args.expname)
    logger.info(f"Arguments: {args}")
    logger.info(f"SIMD MODE: Using 16384 values/ciphertext (16x larger than v16)")
    logger.info(f"Expected speedup: ~16x fewer ciphertexts to encrypt/decrypt/aggregate")
    
    # Get dataset
    trainLoaders, testLoader = get_dataset_HAV(args)
    
    # Initialize model with correct input size
    serverModel = Model(input_size=20531, output_size=len(cancerTypes), hidden_size=args.hidden_size)
    total_params = sum(p.numel() for p in serverModel.parameters())
    logger.info(f"Model: {total_params} total parameters")
    
    # Train
    criterion = nn.NLLLoss()
    device = torch.device(args.device)
    
    use_fhe = not args.no_fhe
    
    start_time = time.time()
    trained_model = Train(
        logger, trainLoaders, testLoader, serverModel, criterion, 
        device, num_epochs=args.epochs, use_fhe=use_fhe, num_clients=args.client
    )
    total_time = time.time() - start_time
    
    # Final test
    logger.info("Final Test:")
    Testing(trained_model, testLoader, criterion, device, logger)
    
    logger.info(f"TOTAL TIME: {total_time:.2f}s")
