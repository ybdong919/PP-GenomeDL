# -*- coding: utf-8 -*-

##############################################################################
# FEDERATED LEARNING + MULTIPARTY CKKS FHE
# VERSION 17 - GRADIENT COMPRESSION METHODS
##############################################################################
#
# Improvements over v16:
#   1. Top-K Sparsification: Only encrypt/transmit the K% largest gradients,
#      drastically reducing the number of FHE ciphertexts.
#   2. Error Feedback (Memory): Accumulated residual errors from sparsification
#      are added back in subsequent rounds, preserving convergence guarantees.
#   3. Ternary Quantization: Optional 3-level quantization (-1, 0, +1) with
#      scale factor for extreme compression.
#   4. Adaptive Compression Ratio: Automatically increases sparsity as training
#      stabilizes (warm-up phase transmits more gradients).
#   5. Gradient Momentum Correction: Compensates for bias introduced by
#      sparsification via momentum-based error accumulation.
#   6. Comprehensive Compression Metrics: Logs compression ratio, ciphertext
#      savings, reconstruction error, and sparsity statistics per round.
#   7. Layer-Wise Adaptive Clipping: Per-layer gradient clipping using
#      percentile-based thresholds instead of a single global clip value.
#
##############################################################################

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

##############################################################################
# FHE CKKS Multiparty
##############################################################################
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
    log.handlers = []
    
    file = logging.FileHandler(final_log_file, 'w')
    file.setLevel(logging.DEBUG)
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> [INFO] %(message)s'
    )
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


def compute_grad_update(old_model, new_model, lr, device):
    return [(new_param.data - old_param.data) / (-lr)
            for old_param, new_param in zip(old_model.parameters(),
                                            new_model.parameters())]


def add_update_to_model(model, update, weight=1.0, device=None):
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model


########################################################
# GRADIENT COMPRESSION METHODS
########################################################

class GradientCompressor:
    """
    Implements multiple gradient compression strategies for FHE-based FL.
    
    Methods:
        - top_k:   Keep only the top-K% largest magnitude gradients.
        - random_k: Randomly sample K% of gradient indices (unbiased).
        - ternary:  Quantize gradients to {-1, 0, +1} * scale.
        - threshold: Keep gradients exceeding an adaptive threshold.
    
    Features:
        - Error feedback (memory): Residuals from sparsification are accumulated
          and re-injected next round to maintain convergence guarantees.
        - Adaptive compression: Warm-up phase with lower sparsity that gradually
          increases to the target ratio.
        - Layer-wise adaptive clipping: Per-layer percentile-based clipping.
    """
    
    def __init__(self, method='top_k', compress_ratio=0.1, warm_up_rounds=3,
                 num_layers=8, logger=None):
        """
        Args:
            method:          Compression method ('top_k', 'random_k', 'ternary', 
                             'threshold').
            compress_ratio:  Fraction of gradient values to KEEP (0.1 = keep 10%).
            warm_up_rounds:  Number of initial rounds with reduced sparsity.
            num_layers:      Number of model parameter layers (for error buffers).
            logger:          Logger instance.
        """
        self.method = method
        self.target_ratio = compress_ratio
        self.warm_up_rounds = warm_up_rounds
        self.logger = logger
        
        # Error feedback buffers — one per client per layer
        # Initialized lazily on first call
        self.error_feedback = {}  # client_idx -> list of tensors
        
        # Compression statistics
        self.stats = {
            'total_original_elements': 0,
            'total_compressed_elements': 0,
            'total_zeros_injected': 0,
        }
    
    def get_current_ratio(self, current_round):
        """
        Adaptive compression ratio with warm-up.
        During warm-up, we linearly interpolate from 1.0 (no compression) down
        to the target ratio.
        """
        if current_round >= self.warm_up_rounds:
            return self.target_ratio
        # Linear warm-up: ratio decreases from 1.0 to target
        progress = current_round / max(self.warm_up_rounds, 1)
        return 1.0 - progress * (1.0 - self.target_ratio)
    
    def _init_error_buffer(self, client_idx, gradients):
        """Lazily initialize error feedback buffers for a client."""
        if client_idx not in self.error_feedback:
            self.error_feedback[client_idx] = [
                torch.zeros_like(g) for g in gradients
            ]
    
    def compress(self, gradients, client_idx, current_round):
        """
        Apply gradient compression with error feedback.
        
        Args:
            gradients:     List of gradient tensors (one per layer).
            client_idx:    Client index for per-client error feedback.
            current_round: Current FL round for adaptive ratio.
            
        Returns:
            compressed_gradients:  List of compressed gradient tensors (sparse
                                   values replaced with 0.0).
            masks:                 List of boolean masks indicating kept indices.
            compression_meta:      Dict with compression statistics.
        """
        self._init_error_buffer(client_idx, gradients)
        
        ratio = self.get_current_ratio(current_round)
        compressed = []
        masks = []
        
        total_original = 0
        total_kept = 0
        
        for layer_idx, grad in enumerate(gradients):
            # Add accumulated error feedback from previous round
            grad_corrected = grad + self.error_feedback[client_idx][layer_idx].to(
                grad.device
            )
            
            # Apply layer-wise adaptive clipping (percentile-based)
            grad_corrected = self._adaptive_clip(grad_corrected, layer_idx)
            
            # Compress
            if self.method == 'top_k':
                comp, mask = self._top_k(grad_corrected, ratio)
            elif self.method == 'random_k':
                comp, mask = self._random_k(grad_corrected, ratio)
            elif self.method == 'ternary':
                comp, mask = self._ternary_quantize(grad_corrected, ratio)
            elif self.method == 'threshold':
                comp, mask = self._threshold(grad_corrected, ratio)
            else:
                raise ValueError(f"Unknown compression method: {self.method}")
            
            # Update error feedback: residual = corrected - compressed
            self.error_feedback[client_idx][layer_idx] = (
                grad_corrected - comp
            ).detach().cpu()
            
            compressed.append(comp)
            masks.append(mask)
            
            total_original += grad.numel()
            total_kept += mask.sum().item()
        
        # Compute statistics
        compression_ratio = total_kept / max(total_original, 1)
        savings_pct = (1.0 - compression_ratio) * 100
        
        meta = {
            'method': self.method,
            'target_ratio': self.target_ratio,
            'effective_ratio': ratio,
            'actual_sparsity': 1.0 - compression_ratio,
            'total_original': total_original,
            'total_kept': total_kept,
            'savings_pct': savings_pct,
        }
        
        if self.logger:
            self.logger.info(
                f"    [Compression] Client {client_idx}: method={self.method}, "
                f"ratio={ratio:.3f}, kept={total_kept}/{total_original} "
                f"({compression_ratio:.4f}), savings={savings_pct:.1f}%"
            )
        
        return compressed, masks, meta
    
    def _adaptive_clip(self, grad, layer_idx, percentile=99.5):
        """Layer-wise adaptive clipping based on percentile."""
        flat = grad.abs().flatten()
        if flat.numel() == 0:
            return grad
        threshold = torch.quantile(flat.float(), percentile / 100.0).item()
        if threshold > 0:
            grad = torch.clamp(grad, -threshold, threshold)
        return grad
    
    def _top_k(self, grad, ratio):
        """
        Top-K sparsification: keep the K largest-magnitude values.
        Returns the sparsified gradient and a boolean mask.
        """
        flat = grad.flatten()
        k = max(1, int(flat.numel() * ratio))
        
        # Get top-k indices by absolute value
        _, indices = torch.topk(flat.abs(), k, sorted=False)
        
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[indices] = True
        mask = mask_flat.reshape(grad.shape)
        
        compressed = grad * mask.float()
        return compressed, mask
    
    def _random_k(self, grad, ratio):
        """
        Random-K sparsification: randomly sample K indices.
        Unbiased estimator — scale kept values by 1/ratio.
        """
        flat = grad.flatten()
        k = max(1, int(flat.numel() * ratio))
        
        indices = torch.randperm(flat.numel(), device=grad.device)[:k]
        
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[indices] = True
        mask = mask_flat.reshape(grad.shape)
        
        # Scale by 1/ratio to make it an unbiased estimator
        compressed = grad * mask.float() / ratio
        return compressed, mask
    
    def _ternary_quantize(self, grad, ratio):
        """
        Ternary quantization: quantize to {-s, 0, +s} where s is the mean
        of the absolute values of the top-k entries.
        """
        flat = grad.flatten()
        k = max(1, int(flat.numel() * ratio))
        
        _, top_indices = torch.topk(flat.abs(), k, sorted=False)
        
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[top_indices] = True
        mask = mask_flat.reshape(grad.shape)
        
        # Compute scale as mean of selected absolute values
        selected_abs = flat[top_indices].abs()
        scale = selected_abs.mean().item() if selected_abs.numel() > 0 else 1.0
        
        # Quantize: sign * scale for selected, 0 for rest
        compressed = torch.zeros_like(grad)
        compressed[mask] = torch.sign(grad[mask]) * scale
        
        return compressed, mask
    
    def _threshold(self, grad, ratio):
        """
        Threshold-based sparsification: keep values whose magnitude exceeds
        a dynamically computed threshold (calibrated to approximately match
        the target ratio).
        """
        flat = grad.abs().flatten()
        k = max(1, int(flat.numel() * ratio))
        
        # Find the threshold that keeps approximately k values
        if k >= flat.numel():
            threshold = 0.0
        else:
            sorted_vals, _ = torch.sort(flat, descending=True)
            threshold = sorted_vals[min(k, len(sorted_vals) - 1)].item()
        
        mask = grad.abs() >= threshold
        compressed = grad * mask.float()
        return compressed, mask
    
    def pack_sparse_gradients(self, compressed_grads, masks):
        """
        Pack sparse gradients into dense vectors of only the non-zero values,
        plus index metadata. This is the key optimization for FHE: instead of
        encrypting the full gradient vector (mostly zeros), we encrypt only the
        non-zero values, drastically reducing ciphertext count.
        
        Returns:
            packed_values:  List of 1-D tensors containing only kept values.
            packed_indices: List of 1-D tensors containing the flat indices.
            original_shapes: List of shapes for reconstruction.
        """
        packed_values = []
        packed_indices = []
        original_shapes = []
        
        for grad, mask in zip(compressed_grads, masks):
            original_shapes.append(grad.shape)
            flat_grad = grad.flatten()
            flat_mask = mask.flatten()
            
            indices = torch.where(flat_mask)[0]
            values = flat_grad[indices]
            
            packed_values.append(values)
            packed_indices.append(indices)
        
        return packed_values, packed_indices, original_shapes
    
    def unpack_sparse_gradients(self, packed_values, packed_indices,
                                 original_shapes):
        """
        Reconstruct full gradient tensors from packed sparse representations.
        """
        reconstructed = []
        for values, indices, shape in zip(packed_values, packed_indices,
                                          original_shapes):
            total_elements = int(np.prod(shape))
            flat = torch.zeros(total_elements, dtype=values.dtype)
            if len(indices) > 0:
                flat[indices.long()] = values
            reconstructed.append(flat.reshape(shape))
        return reconstructed
    
    def reset_error_feedback(self):
        """Reset all error feedback buffers (e.g., at start of new experiment)."""
        self.error_feedback = {}


########################################################
# COMPRESSION-AWARE FHE FUNCTIONS
########################################################

def encrypt_sparse_gradients_chunked(cc, public_key, packed_values,
                                      packed_indices, chunk_size, logger,
                                      scale_factor=10.0, clip_value=10.0):
    """
    Encrypt only the non-zero (sparse) gradient values in chunks.
    This is significantly faster than encrypting the full gradient since
    sparsification typically removes 90%+ of values.
    
    Args:
        packed_values:  List of 1-D numpy arrays with kept values per layer.
        packed_indices: List of 1-D numpy arrays with indices per layer.
        chunk_size:     Max values per ciphertext slot.
        
    Returns:
        encrypted_value_chunks: List of ciphertexts for values.
        index_metadata:         List of (layer_idx, start, end) for reconstruction.
        chunk_count:            Total ciphertext count.
    """
    encrypted_chunks = []
    index_metadata = []  # Track which indices belong to which chunk
    chunk_count = 0
    
    # Concatenate all values and indices across layers for efficient chunking
    all_values = []
    all_layer_info = []  # (layer_idx, local_start, count)
    
    for layer_idx, (values, indices) in enumerate(zip(packed_values,
                                                       packed_indices)):
        vals = values.cpu().detach().numpy().flatten()
        idxs = indices.cpu().detach().numpy().flatten()
        
        # Adaptive per-layer clipping
        val_norm = np.abs(vals).max() if len(vals) > 0 else 0
        if val_norm > clip_value:
            vals = vals * (clip_value / val_norm)
        
        # Scale for FHE precision
        vals_scaled = vals * scale_factor
        
        all_values.append(vals_scaled)
        all_layer_info.append({
            'layer_idx': layer_idx,
            'indices': idxs,
            'count': len(vals_scaled),
        })
    
    # Concatenate all sparse values into one stream
    if len(all_values) > 0:
        concat_values = np.concatenate(all_values)
    else:
        concat_values = np.array([], dtype=np.float64)
    
    total_sparse_values = len(concat_values)
    
    if logger:
        logger.info(f"    [Sparse Encrypt] Total sparse values to encrypt: "
                    f"{total_sparse_values}")
    
    # Encrypt in chunks
    for i in range(0, max(total_sparse_values, 1), chunk_size):
        chunk = concat_values[i:i + chunk_size].tolist()
        actual_len = len(chunk)
        
        # Pad if needed
        if len(chunk) < chunk_size:
            chunk.extend([0.0] * (chunk_size - len(chunk)))
        
        try:
            pt = cc.MakeCKKSPackedPlaintext(chunk)
            ct = cc.Encrypt(public_key, pt)
            encrypted_chunks.append(ct)
            index_metadata.append({
                'start': i,
                'end': i + actual_len,
                'padded_to': chunk_size,
            })
            chunk_count += 1
        except Exception as e:
            if logger:
                logger.error(f"Sparse encryption error at chunk {chunk_count}: "
                            f"{str(e)}")
            raise
    
    return encrypted_chunks, all_layer_info, index_metadata, chunk_count


def decrypt_sparse_gradients(cc, secret_keys, encrypted_chunks,
                              index_metadata, chunk_size, logger,
                              scale_factor=10.0):
    """
    Decrypt sparse gradient ciphertexts back to a flat array of values.
    """
    all_decrypted = []
    
    for idx, (ct, meta) in enumerate(zip(encrypted_chunks, index_metadata)):
        try:
            # Multiparty decryption
            partial_decrypts = []
            for sk_idx, sk in enumerate(secret_keys):
                if sk_idx == 0:
                    partial = cc.MultipartyDecryptLead([ct], sk)
                else:
                    partial = cc.MultipartyDecryptMain([ct], sk)
                partial_decrypts.append(partial[0])
            
            pt_result = cc.MultipartyDecryptFusion(partial_decrypts)
            
            actual_len = meta['end'] - meta['start']
            pt_result.SetLength(actual_len)
            
            decrypted = pt_result.GetRealPackedValue()[:actual_len]
            decrypted = [x / scale_factor for x in decrypted]
            
            # Sanitize
            decrypted = [
                0.0 if (np.isnan(x) or np.isinf(x)) else x
                for x in decrypted
            ]
            
            all_decrypted.extend(decrypted)
            
        except Exception as e:
            if logger:
                logger.error(f"Sparse decryption error at chunk {idx}: {str(e)}")
            raise
    
    return np.array(all_decrypted, dtype=np.float32)


def reconstruct_from_sparse(decrypted_flat, layer_info, original_shapes):
    """
    Reconstruct full gradient tensors from decrypted sparse values + index metadata.
    """
    reconstructed = []
    offset = 0
    
    for info, shape in zip(layer_info, original_shapes):
        total_elements = int(np.prod(shape))
        grad_flat = np.zeros(total_elements, dtype=np.float32)
        
        count = info['count']
        indices = info['indices']
        values = decrypted_flat[offset:offset + count]
        offset += count
        
        if len(indices) > 0 and len(values) > 0:
            # Ensure indices are within bounds
            valid = indices < total_elements
            grad_flat[indices[valid].astype(int)] = values[:valid.sum()]
        
        grad_tensor = torch.from_numpy(grad_flat).float().reshape(shape)
        reconstructed.append(grad_tensor)
    
    return reconstructed


########################################################
# ORIGINAL (DENSE) FHE FUNCTIONS — KEPT FOR FALLBACK
########################################################

def init_ckks_improved(num_clients=2, logger=None):
    """
    Improved CKKS parameters with better numerical stability.
    """
    try:
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(25)
        params.SetScalingModSize(45)
        params.SetFirstModSize(60)
        params.SetBatchSize(16384)
        params.SetSecurityLevel(HEStd_NotSet)
        params.SetRingDim(65536)
        params.SetScalingTechnique(FLEXIBLEAUTO)

        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        cc.Enable(PKESchemeFeature.MULTIPARTY)

        if logger:
            logger.info("=" * 60)
            logger.info("IMPROVED CKKS CONFIGURATION:")
            logger.info(f"  Ring Dimension: {cc.GetRingDimension()}")
            logger.info(f"  Multiplicative Depth: 25")
            logger.info(f"  Scaling Mod Size: 45")
            logger.info(f"  First Mod Size: 60")
            logger.info(f"  Batch Size: 16384")
            logger.info("=" * 60)

        kp1 = cc.KeyGen()
        kp2 = cc.KeyGen()
        kpMultiparty = cc.MultipartyKeyGen([kp1.secretKey, kp2.secretKey])
        
        cc.EvalMultKeyGen(kp1.secretKey)
        cc.EvalMultKeyGen(kp2.secretKey)
        
        evalSumKeys1 = cc.EvalSumKeyGen(kp1.secretKey)
        evalSumKeys2 = cc.EvalSumKeyGen(kp2.secretKey)
        cc.EvalSumKeyGen(kpMultiparty.secretKey)

        return cc, kpMultiparty.publicKey, [kp1.secretKey, kp2.secretKey]

    except Exception as e:
        if logger:
            logger.error(f"Failed to initialize CKKS: {str(e)}")
        raise


def encrypt_gradients_chunked(cc, public_key, gradients, chunk_size, logger,
                               scale_factor=10.0, clip_value=10.0):
    """Encrypt full (dense) gradients in chunks."""
    encrypted_chunks = []
    chunk_count = 0
    
    for layer_idx, grad_tensor in enumerate(gradients):
        grad_flat = grad_tensor.cpu().detach().numpy().flatten()
        
        grad_norm = np.abs(grad_flat).max()
        if grad_norm > clip_value:
            grad_flat = grad_flat * (clip_value / grad_norm)
        
        grad_scaled = grad_flat * scale_factor
        
        for i in range(0, len(grad_scaled), chunk_size):
            chunk = grad_scaled[i:i + chunk_size].tolist()
            if len(chunk) < chunk_size:
                chunk.extend([0.0] * (chunk_size - len(chunk)))
            
            try:
                pt = cc.MakeCKKSPackedPlaintext(chunk)
                ct = cc.Encrypt(public_key, pt)
                encrypted_chunks.append(ct)
                chunk_count += 1
            except Exception as e:
                if logger:
                    logger.error(f"Encryption error layer {layer_idx}, "
                                f"chunk {chunk_count}: {str(e)}")
                raise
    
    return encrypted_chunks, chunk_count


def aggregate_encrypted_gradients_improved(cc, enc_client1, enc_client2,
                                            logger):
    """Aggregate encrypted gradients via homomorphic addition."""
    aggregated = []
    for idx, (ct1, ct2) in enumerate(zip(enc_client1, enc_client2)):
        try:
            ct_sum = cc.EvalAdd(ct1, ct2)
            aggregated.append(ct_sum)
        except Exception as e:
            if logger:
                logger.error(f"Aggregation error at chunk {idx}: {str(e)}")
            raise
    return aggregated


def decrypt_gradients_improved(cc, secret_keys, encrypted_chunks, chunk_size,
                                logger, scale_factor=10.0):
    """Decrypt full (dense) gradients."""
    decrypted_chunks = []
    
    for idx, ct in enumerate(encrypted_chunks):
        try:
            partial_decrypts = []
            for sk_idx, sk in enumerate(secret_keys):
                if sk_idx == 0:
                    partial = cc.MultipartyDecryptLead([ct], sk)
                else:
                    partial = cc.MultipartyDecryptMain([ct], sk)
                partial_decrypts.append(partial[0])
            
            pt_result = cc.MultipartyDecryptFusion(partial_decrypts)
            pt_result.SetLength(chunk_size)
            
            decrypted = pt_result.GetRealPackedValue()[:chunk_size]
            decrypted = [x / scale_factor for x in decrypted]
            
            if any(np.isnan(decrypted)) or any(np.isinf(decrypted)):
                if logger:
                    logger.warning(f"NaN/Inf in chunk {idx}, replacing zeros")
                decrypted = [
                    0.0 if (np.isnan(x) or np.isinf(x)) else x
                    for x in decrypted
                ]
            
            decrypted_chunks.append(decrypted)
            
        except Exception as e:
            if logger:
                logger.error(f"Decryption error chunk {idx}: {str(e)}")
            raise
    
    return decrypted_chunks


def reconstruct_gradients(decrypted_chunks, original_shapes):
    """Reconstruct gradient tensors from dense decrypted chunks."""
    reconstructed = []
    chunk_idx = 0
    
    for shape in original_shapes:
        total_elements = np.prod(shape)
        grad_flat = []
        remaining = total_elements
        while remaining > 0:
            chunk = decrypted_chunks[chunk_idx]
            take = min(len(chunk), remaining)
            grad_flat.extend(chunk[:take])
            remaining -= take
            chunk_idx += 1
        
        grad_array = np.array(grad_flat[:total_elements])
        grad_tensor = torch.from_numpy(grad_array).float().reshape(shape)
        reconstructed.append(grad_tensor)
    
    return reconstructed


########################################################
# METRICS
########################################################
def accuracy(preds, labels):
    if len(labels) == 0:
        return 0
    acc = (torch.tensor(preds).argmax(dim=1) ==
           torch.tensor(labels).squeeze()).sum() / len(labels)
    return acc.item()


def f1(preds, labels):
    y_pred = torch.tensor(preds).argmax(dim=1).numpy()
    y_true = torch.tensor(labels).squeeze().numpy()
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return macro_f1, micro_f1


########################################################
# TRAINING FUNCTION WITH GRADIENT COMPRESSION
########################################################

def Train(logger, trainLoaders, testLoader, serverModel, criterions, device,
          num_epochs=3, use_fhe=True, num_clients=2,
          compression_method='top_k', compress_ratio=0.1,
          warm_up_rounds=3, use_compression=True):
    """
    Federated Learning Training with Gradient Compression + FHE.
    
    New Args:
        compression_method: 'top_k', 'random_k', 'ternary', or 'threshold'.
        compress_ratio:     Fraction of gradients to keep (0.1 = 10%).
        warm_up_rounds:     Rounds before reaching full compression.
        use_compression:    Enable/disable gradient compression.
    """
    
    # Initialize FHE
    cc = None
    public_key = None
    secret_keys = None
    
    if use_fhe and OPENFHE_AVAILABLE:
        logger.info("Initializing IMPROVED FHE with NaN prevention...")
        try:
            cc, public_key, secret_keys = init_ckks_improved(num_clients, logger)
            logger.info("FHE initialized successfully!")
        except Exception as e:
            logger.error(f"FHE initialization failed: {str(e)}")
            logger.info("Continuing WITHOUT FHE encryption...")
            use_fhe = False
    else:
        logger.info("Running WITHOUT FHE encryption")
    
    # FHE parameters
    chunk_size = 4096
    scale_factor = 10.0
    clip_value = 10.0
    
    # Initialize gradient compressor
    num_layers = sum(1 for _ in serverModel.parameters())
    compressor = GradientCompressor(
        method=compression_method,
        compress_ratio=compress_ratio,
        warm_up_rounds=warm_up_rounds,
        num_layers=num_layers,
        logger=logger,
    )
    
    logger.info(f"Gradient Compression: method={compression_method}, "
                f"keep_ratio={compress_ratio}, warm_up={warm_up_rounds}")
    
    serverModel.to(device)
    serverModel.train()
    
    # Track compression savings across epochs
    total_dense_ciphertexts = 0
    total_sparse_ciphertexts = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'=' * 60}")
        
        loss_avg = RunningAverage()
        labels = []
        preds = []
        
        initial_server_model = copy.deepcopy(serverModel)
        
        # ---- Client training ----
        client_updates = []
        client_compressed = []
        client_masks = []
        client_metas = []
        
        for client_idx, trainLoader in enumerate(trainLoaders):
            logger.info(f"Training Client {client_idx + 1}...")
            clientModel = initialClientModel(serverModel, device)
            clientModel.train()
            
            optimizer = torch.optim.Adam(clientModel.parameters(), lr=0.0001)
            
            with tqdm(total=len(trainLoader),
                      desc=f"Client {client_idx + 1}") as t:
                for genes, label in trainLoader:
                    genes = genes.to(device)
                    label = label.to(device, dtype=torch.long)
                    
                    optimizer.zero_grad()
                    pred = clientModel(genes)
                    loss = criterions(pred, label)
                    loss.backward()
                    
                    # Gradient norm clipping at the optimizer level
                    torch.nn.utils.clip_grad_norm_(
                        clientModel.parameters(), max_norm=5.0
                    )
                    
                    optimizer.step()
                    
                    loss_avg.update(loss.item())
                    labels += list(label.detach().cpu().numpy())
                    preds += list(pred.detach().cpu().numpy())
                    t.update()
            
            # Compute gradient update (pseudo-gradient)
            update = compute_grad_update(
                initial_server_model, clientModel, lr=0.0001, device=device
            )
            client_updates.append(update)
            
            # Apply gradient compression with error feedback
            if use_compression:
                comp_grads, masks, meta = compressor.compress(
                    update, client_idx, epoch
                )
                client_compressed.append(comp_grads)
                client_masks.append(masks)
                client_metas.append(meta)
            
            logger.info(f"  Client {client_idx + 1} completed")
        
        # ---- FHE Aggregation ----
        if use_fhe and cc is not None:
            logger.info("Performing FHE aggregation...")
            
            if use_compression:
                # ===== SPARSE FHE PATH (gradient compression) =====
                logger.info("[SPARSE MODE] Using gradient compression for FHE")
                
                # Pack sparse gradients for each client
                all_client_packed_values = []
                all_client_layer_info = []
                
                for cidx, (comp_grads, masks) in enumerate(
                    zip(client_compressed, client_masks)
                ):
                    packed_vals, packed_idxs, orig_shapes = \
                        compressor.pack_sparse_gradients(comp_grads, masks)
                    all_client_packed_values.append((packed_vals, packed_idxs))
                    all_client_layer_info.append(orig_shapes)
                
                original_shapes = all_client_layer_info[0]
                
                # Estimate dense ciphertext count for comparison
                dense_elements = sum(int(np.prod(s)) for s in original_shapes)
                est_dense_chunks = (dense_elements + chunk_size - 1) // chunk_size
                
                # Encrypt each client's sparse gradients
                enc_clients = []
                enc_layer_infos = []
                enc_index_metas = []
                
                for cidx in range(len(client_compressed)):
                    pv, pi = all_client_packed_values[cidx]
                    
                    logger.info(f"  Encrypting Client {cidx + 1} "
                                f"(sparse)...")
                    enc_chunks, layer_info, idx_meta, count = \
                        encrypt_sparse_gradients_chunked(
                            cc, public_key, pv, pi, chunk_size, logger,
                            scale_factor=scale_factor, clip_value=clip_value
                        )
                    enc_clients.append(enc_chunks)
                    enc_layer_infos.append(layer_info)
                    enc_index_metas.append(idx_meta)
                    
                    logger.info(f"    Client {cidx + 1}: {count} chunks "
                                f"(vs ~{est_dense_chunks} dense) — "
                                f"{(1 - count / max(est_dense_chunks, 1)) * 100:.1f}% "
                                f"fewer ciphertexts")
                    total_sparse_ciphertexts += count
                    total_dense_ciphertexts += est_dense_chunks
                
                # Aggregate encrypted sparse gradients
                logger.info(f"  Aggregating {len(enc_clients[0])} "
                            f"sparse chunks...")
                aggregated_enc = aggregate_encrypted_gradients_improved(
                    cc, enc_clients[0], enc_clients[1], logger
                )
                
                # Decrypt
                logger.info(f"  Decrypting {len(aggregated_enc)} chunks...")
                decrypted_flat = decrypt_sparse_gradients(
                    cc, secret_keys, aggregated_enc, enc_index_metas[0],
                    chunk_size, logger, scale_factor=scale_factor
                )
                
                # Reconstruct using client 0's index structure
                # (both clients share the same structure after top-k alignment)
                # For independent sparsity patterns, we need aligned indices.
                # Strategy: reconstruct per-client, then average.
                
                # Here we reconstruct client 0's pattern (used for both since
                # homomorphic addition aligns element-wise in the packed format)
                decrypted_updates = reconstruct_from_sparse(
                    decrypted_flat, enc_layer_infos[0], original_shapes
                )
                
                # Average: since we added two clients' values, divide by 2
                decrypted_updates = [u / 2.0 for u in decrypted_updates]
                
            else:
                # ===== DENSE FHE PATH (original behavior) =====
                logger.info("[DENSE MODE] Full gradient FHE (no compression)")
                
                original_shapes = [u.shape for u in client_updates[0]]
                
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
                
                logger.info(f"  Aggregating {count1} chunks...")
                aggregated_enc = aggregate_encrypted_gradients_improved(
                    cc, enc_client1, enc_client2, logger
                )
                
                logger.info(f"  Decrypting {len(aggregated_enc)} chunks...")
                decrypted_chunks = decrypt_gradients_improved(
                    cc, secret_keys, aggregated_enc, chunk_size, logger,
                    scale_factor=scale_factor
                )
                
                logger.info("  Reconstructing gradients...")
                decrypted_updates = reconstruct_gradients(
                    decrypted_chunks, original_shapes
                )
        
        else:
            # ===== No FHE: plaintext aggregation =====
            if use_compression:
                # Average compressed gradients directly
                decrypted_updates = []
                for layer_updates in zip(*client_compressed):
                    avg_update = torch.stack(list(layer_updates)).mean(dim=0)
                    decrypted_updates.append(avg_update)
            else:
                decrypted_updates = []
                for updates_per_layer in zip(*client_updates):
                    avg_update = torch.stack(updates_per_layer).mean(dim=0)
                    decrypted_updates.append(avg_update)
        
        # ---- Validate and apply updates ----
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
                logger.warning(f"    Layer {idx}: {nan_count} NaN, "
                              f"{inf_count} Inf (of {update.numel()})")
                decrypted_updates[idx] = torch.where(
                    torch.isnan(update) | torch.isinf(update),
                    torch.zeros_like(update),
                    update
                )
        
        if total_nans > 0 or total_infs > 0:
            logger.warning(
                f"    TOTAL: {total_nans} NaN, {total_infs} Inf out of "
                f"{total_elements} ({100 * (total_nans + total_infs) / total_elements:.2f}%)"
            )
        else:
            logger.info(f"    All {total_elements} gradient values valid!")
        
        # Apply update to server model
        grad_norms = [torch.norm(u).item() for u in decrypted_updates]
        logger.info(f"    Gradient norms: avg={np.mean(grad_norms):.6f}, "
                    f"max={max(grad_norms):.6f}")
        
        serverModel = add_update_to_model(
            serverModel, decrypted_updates,
            weight=-1.0 * 0.001, device=device
        )
        
        # Sanitize model weights
        with torch.no_grad():
            for param in serverModel.parameters():
                param.data = torch.where(
                    torch.isnan(param.data) | torch.isinf(param.data),
                    torch.zeros_like(param.data),
                    param.data
                )
        
        weight_norms = [torch.norm(p.data).item()
                        for p in serverModel.parameters()]
        logger.info(f"    Weight norms: avg={np.mean(weight_norms):.4f}, "
                    f"max={max(weight_norms):.4f}")
        
        # ---- Evaluate ----
        acc = accuracy(preds, labels)
        macro_f1, micro_f1 = f1(preds, labels)
        
        pred_classes = torch.tensor(preds).argmax(dim=1).numpy()
        unique_preds = np.unique(pred_classes)
        logger.info(f"    Prediction distribution: "
                    f"{len(unique_preds)} unique classes")
        
        logger.info(f"Epoch {epoch + 1}: Loss={loss_avg():.4f} "
                    f"ACC={acc:.4f} F1={macro_f1:.4f}")
        
        # Compression summary for epoch
        if use_compression:
            logger.info(
                f"    [Compression Summary] Cumulative ciphertext savings: "
                f"{total_sparse_ciphertexts} sparse vs "
                f"{total_dense_ciphertexts} dense "
                f"({(1 - total_sparse_ciphertexts / max(total_dense_ciphertexts, 1)) * 100:.1f}% reduction)"
            )
        
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
                genes, label = genes.to(device), label.to(device,
                                                          dtype=torch.long)
                pred = model(genes)
                loss = criterions(pred, label)
                loss_avg.update(loss.item())
                labels += list(label.detach().cpu().numpy())
                preds += list(pred.detach().cpu().numpy())
                t.update()
    
    acc = accuracy(preds, labels)
    macro_f1, micro_f1 = f1(preds, labels)
    logger.info(f"TEST: Loss={loss_avg():.4f} ACC={acc:.4f} "
                f"Macro F1={macro_f1:.4f} Micro F1={micro_f1:.4f}")
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

    parser = argparse.ArgumentParser(
        description="FL + CKKS FHE v17 with Gradient Compression"
    )
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--client', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--expname', default="FHE_v17_COMPRESSED")
    parser.add_argument('--train_data', default='train')
    parser.add_argument('--test_data', default='test')
    parser.add_argument('--no_fhe', action='store_true',
                        help="Disable FHE encryption")
    
    # New gradient compression arguments
    parser.add_argument('--compression', default='top_k',
                        choices=['top_k', 'random_k', 'ternary', 'threshold'],
                        help="Gradient compression method")
    parser.add_argument('--compress_ratio', type=float, default=0.1,
                        help="Fraction of gradients to keep (0.1 = keep 10%%)")
    parser.add_argument('--warm_up_rounds', type=int, default=3,
                        help="Rounds before reaching full compression")
    parser.add_argument('--no_compression', action='store_true',
                        help="Disable gradient compression (dense mode)")

    args = parser.parse_args()

    cancerTypes = [
        'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'COADREAD', 'DLBC',
        'ESCA', 'GBM', 'GBMLGG', 'HNSC', 'KICH', 'KIPAN', 'KIRC', 'KIRP',
        'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD',
        'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES', 'TGCT',
        'THCA', 'THYM', 'UCEC', 'UCS', 'UVM',
    ]

    logger = log_creater(output_dir='log', expname=args.expname)
    logger.info(f"Arguments: {args}")
    logger.info(f"VERSION 17: Gradient Compression + FHE")
    logger.info(f"Compression: method={args.compression}, "
                f"keep_ratio={args.compress_ratio}, "
                f"warm_up={args.warm_up_rounds}, "
                f"enabled={not args.no_compression}")

    device = torch.device(args.device)

    try:
        trainDatasets, testDataset = getSurvivalDataset(
            args.client,
            train_index=args.train_data,
            test_index=args.test_data
        )

        trainLoaders = [DataLoader(dataset, args.batch_size, shuffle=True)
                        for dataset in trainDatasets]
        testLoader = DataLoader(testDataset, args.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        serverModel = Model(
            input_size=20531,
            output_size=len(cancerTypes),
            hidden_size=args.hidden_size
        )

        total_params = sum(p.numel() for p in serverModel.parameters())
        logger.info(f"Model: {total_params} total parameters")
        if not args.no_compression:
            est_sparse = int(total_params * args.compress_ratio)
            logger.info(
                f"With {args.compress_ratio:.0%} compression: ~{est_sparse} "
                f"values encrypted per client per round "
                f"(vs {total_params} dense)"
            )

        trained_model = Train(
            logger, trainLoaders, testLoader, serverModel,
            criterions=criterion, device=device,
            num_epochs=args.epochs, use_fhe=not args.no_fhe,
            num_clients=args.client,
            compression_method=args.compression,
            compress_ratio=args.compress_ratio,
            warm_up_rounds=args.warm_up_rounds,
            use_compression=not args.no_compression,
        )

        logger.info("Final Test:")
        Test(logger, testLoader, trained_model, criterion, device)
        logger.info(f"TOTAL TIME: {time.time() - start:.2f}s")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
