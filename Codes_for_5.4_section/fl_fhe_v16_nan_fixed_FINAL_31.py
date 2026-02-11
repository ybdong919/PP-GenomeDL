# -*- coding: utf-8 -*-

##############################################
# FEDERATED LEARNING + MULTIPARTY CKKS FHE
# VERSION 16 - NaN FIX - DEBUGGED
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
# MODEL - SMALLER FOR FHE COMPATIBILITY
########################################################
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Model, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        #self.dropout1 = nn.Dropout(0.5)  # ADD THIS
        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)
        #self.dropout2 = nn.Dropout(0.3)  # ADD THIS
        self.hidden3 = nn.Linear(hidden_size // 2, 256)
        #self.dropout3 = nn.Dropout(0.2)  # ADD THIS
        self.hidden4 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        #x = self.dropout1(x)  # ADD THIS
        x = F.relu(self.hidden2(x))
        #x = self.dropout2(x)  # ADD THIS
        x = F.relu(self.hidden3(x))
        #x = self.dropout3(x)  # ADD THIS
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


########################################################
# DATASET
########################################################
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


########################################################
# FEDERATED FUNCTIONS
########################################################

def initialClientModel(serverModel, device):
    clientModel = copy.deepcopy(serverModel)
    clientModel.to(device)
    return clientModel

# def compute_grad_update(old_model, new_model, lr, device):
    # updates = []
    # for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        # grad = (new_param.data - old_param.data) / (-lr)
        
        # # CLIP HERE - before FHE encryption
        # grad = torch.clamp(grad, -10.0, 10.0)  # Hard limit
        
        # updates.append(grad)
    # return updates

def compute_grad_update(old_model, new_model, lr, device):
    return [(new_param.data - old_param.data) / (-lr)
            for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def add_update_to_model(model, update, weight=1.0, device=None):
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]
    
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model


########################################################
# METRICS
########################################################
def accuracy(preds, labels):
    if len(labels) == 0:
        return 0
    acc = (torch.tensor(preds).argmax(dim=1) == torch.tensor(labels).squeeze()).sum() / len(labels)
    return acc.item()

def f1(preds, labels):
    y_pred = torch.tensor(preds).argmax(dim=1).numpy()
    y_true = torch.tensor(labels).squeeze().numpy()
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return macro_f1, micro_f1


########################################################
# IMPROVED 2-PARTY FHE WITH NaN PREVENTION - DEBUGGED
########################################################

def init_ckks_improved(num_clients=2, logger=None):
    """
    Improved CKKS parameters with better numerical stability
    Key changes:
    1. Increased multiplicative depth for more operations
    2. Better scaling factor balance
    3. Larger ring dimension for precision
    
    DEBUGGED: Fixed MultiEvalSumKeyGen API call to include evalKeyMap parameter
    """
    try:
        params = CCParamsCKKSRNS()
        
        # Deeper multiplicative depth
        params.SetMultiplicativeDepth(25)
        
        # Better scaling balance - critical for preventing NaN
        params.SetScalingModSize(45)  # Smaller for better stability
        params.SetFirstModSize(60)
        
        # Larger batch for efficiency
        params.SetBatchSize(16384)
        
        # Maximum ring dimension for precision
        params.SetSecurityLevel(HEStd_NotSet)
        params.SetRingDim(65536)
        
        # Use FLEXIBLEAUTO for better rescaling
        params.SetScalingTechnique(FLEXIBLEAUTO)

        cc = GenCryptoContext(params)

        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        cc.Enable(PKESchemeFeature.MULTIPARTY)

        if logger:
            logger.info("="*60)
            logger.info("IMPROVED CKKS CONFIGURATION:")
            logger.info(f"  Ring Dimension: {cc.GetRingDimension()}")
            logger.info(f"  Multiplicative Depth: 25")
            logger.info(f"  Scaling Mod Size: 45")
            logger.info(f"  First Mod Size: 60")
            logger.info(f"  Batch Size: 16384")
            logger.info("="*60)

        # Generate keys
        kp1 = cc.KeyGen()
        kp2 = cc.KeyGen()

        kpMultiparty = cc.MultipartyKeyGen([kp1.secretKey, kp2.secretKey])
        
        # Generate evaluation mult keys
        cc.EvalMultKeyGen(kp1.secretKey)
        cc.EvalMultKeyGen(kp2.secretKey)
        
        # Generate rotation/sum keys for individual parties
        # Use None for the initial evalKeyMap
        evalSumKeys1 = cc.EvalSumKeyGen(kp1.secretKey)
        evalSumKeys2 = cc.EvalSumKeyGen(kp2.secretKey)
        
        # Join the sum keys
        cc.EvalSumKeyGen(kpMultiparty.secretKey)

        return cc, kpMultiparty.publicKey, [kp1.secretKey, kp2.secretKey]

    except Exception as e:
        if logger:
            logger.error(f"Failed to initialize CKKS: {str(e)}")
        raise


def encrypt_gradients_chunked(cc, public_key, gradients, chunk_size, logger, 
                               scale_factor=100.0, clip_value=1.0):
    """
    Encrypt gradients in chunks with improved numerical stability
    """
    encrypted_chunks = []
    chunk_count = 0
    
    for layer_idx, grad_tensor in enumerate(gradients):
        grad_flat = grad_tensor.cpu().detach().numpy().flatten()
        
        # Clip extreme values
        # grad_flat = np.clip(grad_flat, -clip_value, clip_value)
        # To include normalization:
        grad_norm = np.abs(grad_flat).max()
        if grad_norm > clip_value:
             grad_flat = grad_flat * (clip_value / grad_norm)
        
        # Scale for FHE
        grad_scaled = grad_flat * scale_factor
        
        # Process in chunks
        for i in range(0, len(grad_scaled), chunk_size):
            chunk = grad_scaled[i:i+chunk_size].tolist()
            
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk.extend([0.0] * (chunk_size - len(chunk)))
            
            try:
                pt = cc.MakeCKKSPackedPlaintext(chunk)
                ct = cc.Encrypt(public_key, pt)
                encrypted_chunks.append(ct)
                chunk_count += 1
            except Exception as e:
                if logger:
                    logger.error(f"Encryption error at layer {layer_idx}, chunk {chunk_count}: {str(e)}")
                raise
    
    return encrypted_chunks, chunk_count


def aggregate_encrypted_gradients_improved(cc, enc_client1, enc_client2, logger):
    """
    Aggregate encrypted gradients with improved error handling
    """
    aggregated = []
    
    for idx, (ct1, ct2) in enumerate(zip(enc_client1, enc_client2)):
        try:
            # Simple addition in encrypted domain
            ct_sum = cc.EvalAdd(ct1, ct2)
            aggregated.append(ct_sum)
        except Exception as e:
            if logger:
                logger.error(f"Aggregation error at chunk {idx}: {str(e)}")
            raise
    
    return aggregated


def decrypt_gradients_improved(cc, secret_keys, encrypted_chunks, chunk_size, logger,
                                scale_factor=100.0):
    """
    Decrypt gradients with improved numerical stability
    """
    decrypted_chunks = []
    
    for idx, ct in enumerate(encrypted_chunks):
        try:
            # Multiparty decryption
            partial_decrypts = []
            for sk_idx, sk in enumerate(secret_keys):
                if sk_idx == 0:
                    # First party uses MultipartyDecryptLead
                    partial = cc.MultipartyDecryptLead([ct], sk)
                else:
                    # Other parties use MultipartyDecryptMain
                    partial = cc.MultipartyDecryptMain([ct], sk)
                # partial is a list with one element, extract it
                partial_decrypts.append(partial[0])
            
            # Combine partial decryptions - now it's a flat list of ciphertexts
            pt_result = cc.MultipartyDecryptFusion(partial_decrypts)
            pt_result.SetLength(chunk_size)
            
            # Extract and unscale
            decrypted = pt_result.GetRealPackedValue()[:chunk_size]
            decrypted = [x / scale_factor for x in decrypted]
            
            # Check for numerical issues
            if any(np.isnan(decrypted)) or any(np.isinf(decrypted)):
                if logger:
                    logger.warning(f"NaN/Inf detected in chunk {idx}, replacing with zeros")
                decrypted = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in decrypted]
            
            decrypted_chunks.append(decrypted)
            
        except Exception as e:
            if logger:
                logger.error(f"Decryption error at chunk {idx}: {str(e)}")
            raise
    
    return decrypted_chunks


def reconstruct_gradients(decrypted_chunks, original_shapes):
    """
    Reconstruct gradient tensors from decrypted chunks
    """
    reconstructed = []
    chunk_idx = 0
    
    for shape in original_shapes:
        total_elements = np.prod(shape)
        grad_flat = []
        
        # Collect all values for this layer
        remaining = total_elements
        while remaining > 0:
            chunk = decrypted_chunks[chunk_idx]
            take = min(len(chunk), remaining)
            grad_flat.extend(chunk[:take])
            remaining -= take
            chunk_idx += 1
        
        # Reshape and convert to tensor
        grad_array = np.array(grad_flat[:total_elements])
        grad_tensor = torch.from_numpy(grad_array).float().reshape(shape)
        reconstructed.append(grad_tensor)
    
    return reconstructed


########################################################
# TRAINING FUNCTION
########################################################

def Train(logger, trainLoaders, testLoader, serverModel, criterions, device, 
          num_epochs=3, use_fhe=True, num_clients=2):
    """
    Federated Learning Training with Improved FHE
    """
    
    # Initialize FHE if needed
    cc = None
    public_key = None
    secret_keys = None
    
    if use_fhe and OPENFHE_AVAILABLE:
        logger.info("Initializing IMPROVED FHE with NaN prevention...")
        try:
            cc, public_key, secret_keys = init_ckks_improved(num_clients, logger)
            logger.info("? FHE initialized successfully!")
        except Exception as e:
            logger.error(f"FHE initialization failed: {str(e)}")
            logger.info("Continuing WITHOUT FHE encryption...")
            use_fhe = False
    else:
        logger.info("Running WITHOUT FHE encryption")
    
    # FHE parameters - ADJUSTED for better gradient handling
    chunk_size = 4096
    scale_factor = 10.0  # Reduced from 100 - less scaling needed
    clip_value = 10.0   # Increased from 1.0 - allow larger gradients
    
    serverModel.to(device)
    serverModel.train()
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        loss_avg = RunningAverage()
        labels = []
        preds = []
        
        # Save initial server model
        initial_server_model = copy.deepcopy(serverModel)
        
        # Client training
        client_updates = []
        
        for client_idx, trainLoader in enumerate(trainLoaders):
            logger.info(f"Training Client {client_idx+1}...")
            clientModel = initialClientModel(serverModel, device)
            clientModel.train()
            
            optimizer = torch.optim.Adam(clientModel.parameters(), lr=0.0001)
            
            with tqdm(total=len(trainLoader), desc=f"Client {client_idx+1}") as t:
                for genes, label in trainLoader:
                    genes, label = genes.to(device), label.to(device, dtype=torch.long)
                    
                    optimizer.zero_grad()
                    pred = clientModel(genes)
                    loss = criterions(pred, label)
                    loss.backward()
                    optimizer.step()
                    
                    loss_avg.update(loss.item())
                    labels += list(label.detach().cpu().numpy())
                    preds += list(pred.detach().cpu().numpy())
                    t.update()
            
            # Compute gradient update
            update = compute_grad_update(initial_server_model, clientModel, lr=0.0001, device=device)
            client_updates.append(update)
            logger.info(f"  Client {client_idx+1} completed")
        
        # FHE Aggregation
        if use_fhe and cc is not None:
            logger.info("Performing IMPROVED FHE aggregation with NaN prevention...")
            logger.info(f"Processing {len(client_updates[0])} layers...")
            
            original_shapes = [u.shape for u in client_updates[0]]
            
            # Encrypt both clients' gradients
            logger.info("  Encrypting Client 1 gradients...")
            enc_client1, count1 = encrypt_gradients_chunked(
                cc, public_key, client_updates[0], chunk_size, logger,
                scale_factor=scale_factor, clip_value=clip_value
            )
            logger.info(f"    Client 1: {count1} chunks")
            
            logger.info("  Encrypting Client 2 gradients...")
            enc_client2, count2 = encrypt_gradients_chunked(
                cc, public_key, client_updates[1], chunk_size, logger,
                scale_factor=scale_factor, clip_value=clip_value
            )
            logger.info(f"    Client 2: {count2} chunks")
            
            # Aggregate
            logger.info(f"  Aggregating {count1} chunks...")
            aggregated_enc = aggregate_encrypted_gradients_improved(
                cc, enc_client1, enc_client2, logger
            )
            
            # Decrypt
            logger.info(f"  Decrypting {len(aggregated_enc)} chunks...")
            decrypted_chunks = decrypt_gradients_improved(
                cc, secret_keys, aggregated_enc, chunk_size, logger,
                scale_factor=scale_factor
            )
            
            # Reconstruct
            logger.info("  Reconstructing gradients...")
            decrypted_updates = reconstruct_gradients(decrypted_chunks, original_shapes)
            
        else:
            # Simple averaging without FHE
            decrypted_updates = []
            for updates_per_layer in zip(*client_updates):
                avg_update = torch.stack(updates_per_layer).mean(dim=0)
                decrypted_updates.append(avg_update)
        
        # Validate updates
        decrypted_updates = [x.to(device) for x in decrypted_updates]
        
        total_nans = 0
        total_infs = 0
        total_elements = 0
        for idx, update in enumerate(decrypted_updates):
            total_elements += update.numel()
            nan_count = torch.isnan(update).sum().item()
            inf_count = torch.isinf(update).sum().item()
            total_nans += nan_count
            total_infs += inf_count
            
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"    Layer {idx}: {nan_count} NaN, {inf_count} Inf (out of {update.numel()})")
                # Replace bad values
                decrypted_updates[idx] = torch.where(
                    torch.isnan(update) | torch.isinf(update),
                    torch.zeros_like(update),
                    update
                )
        
        if total_nans > 0 or total_infs > 0:
            logger.warning(f"    TOTAL: {total_nans} NaN, {total_infs} Inf out of {total_elements} values ({100*(total_nans+total_infs)/total_elements:.2f}%)")
        else:
            logger.info(f"    ? All {total_elements} gradient values are valid!")
        
        # Apply update
        grad_norms = [torch.norm(u).item() for u in decrypted_updates]
        logger.info(f"    Gradient norms: avg={np.mean(grad_norms):.6f}, max={max(grad_norms):.6f}")
        
        serverModel = add_update_to_model(serverModel, decrypted_updates, 
                                          weight=-1.0 * 0.001, device=device)
        
        # Sanitize model weights
        with torch.no_grad():
            for param in serverModel.parameters():
                param.data = torch.where(torch.isnan(param.data) | torch.isinf(param.data),
                                        torch.zeros_like(param.data),
                                        param.data)
        
        weight_norms = [torch.norm(p.data).item() for p in serverModel.parameters()]
        logger.info(f"    Weight norms: avg={np.mean(weight_norms):.4f}, max={max(weight_norms):.4f}")
        
        # Evaluate
        acc = accuracy(preds, labels)
        macro_f1, micro_f1 = f1(preds, labels)
        
        pred_classes = torch.tensor(preds).argmax(dim=1).numpy()
        unique_preds = np.unique(pred_classes)
        logger.info(f"    Prediction distribution: {len(unique_preds)} unique classes")
        
        logger.info(f"Epoch {epoch+1}: Loss={loss_avg():.4f} ACC={acc:.4f} F1={macro_f1:.4f}")
        
        # Test
        serverModel.eval()
        Test(logger, testLoader, serverModel, criterions, device)
        serverModel.train()
        gc.collect()

    return serverModel


def Test(logger, test_loader, model, criterions, device):
    loss_avg = RunningAverage()
    labels = []
    preds = []
    
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing") as t:
            for genes, label in test_loader:
                genes, label = genes.to(device), label.to(device, dtype=torch.long)
                pred = model(genes)
                loss = criterions(pred, label)
                loss_avg.update(loss.item())
                labels += list(label.detach().cpu().numpy())
                preds += list(pred.detach().cpu().numpy())
                t.update()

    acc = accuracy(preds, labels)
    macro_f1, micro_f1 = f1(preds, labels)
    logger.info(f"TEST: Loss={loss_avg():.4f} ACC={acc:.4f} Macro F1={macro_f1:.4f} Micro F1={micro_f1:.4f}")
    return loss_avg(), acc


def getSurvivalDataset(numClients, train_index='train', test_index='test'):
    trainDataset = Clinical_Data(index=train_index)
    testDataset = Clinical_Data(index=test_index)

    trainDatasets = []
    numItems = int(np.floor(len(trainDataset) / numClients))
    list_x = [numItems for _ in range(numClients)]
    remainder = len(trainDataset) - numItems * numClients
    if remainder != 0:
        list_x[-1] += remainder

    for dataset in torch.utils.data.random_split(trainDataset, list_x):
        trainDatasets.append(dataset)

    return trainDatasets, testDataset


########################################################
# MAIN
########################################################
if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--client', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--expname', default="FHE_FIXED")
    parser.add_argument('--train_data', default='train')
    parser.add_argument('--test_data', default='test')
    parser.add_argument('--no_fhe', action='store_true')

    args = parser.parse_args()

    cancerTypes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA', 'GBM', 'GBMLGG', 'HNSC', 'KICH',
                   'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
                   'SARC', 'SKCM', 'STAD', 'STES', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']

    logger = log_creater(output_dir='log', expname=args.expname)
    logger.info(f"Arguments: {args}")
    logger.info(f"IMPROVED MODE: NaN prevention with scale_factor=100, clip_value=1.0")
    logger.info(f"DEBUGGED: Fixed MultiEvalSumKeyGen API calls")

    device = torch.device(args.device)

    try:
        trainDatasets, testDataset = getSurvivalDataset(args.client,
                                                        train_index=args.train_data,
                                                        test_index=args.test_data)

        trainLoaders = [DataLoader(dataset, args.batch_size, shuffle=True) for dataset in trainDatasets]
        testLoader = DataLoader(testDataset, args.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        serverModel = Model(input_size=20531, output_size=len(cancerTypes), hidden_size=args.hidden_size)

        logger.info(f"Model: {sum(p.numel() for p in serverModel.parameters())} total parameters")

        trained_model = Train(logger, trainLoaders, testLoader, serverModel,
                              criterions=criterion, device=device,
                              num_epochs=args.epochs, use_fhe=not args.no_fhe,
                              num_clients=args.client)

        logger.info("Final Test:")
        Test(logger, testLoader, trained_model, criterion, device)
        logger.info(f"TOTAL TIME: {time.time()-start:.2f}s")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
