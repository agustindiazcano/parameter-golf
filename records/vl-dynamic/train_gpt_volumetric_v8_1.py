"""
MODIFICACIÓN V8.1 (VERSIÓN 1024 BPE): Cuantización Adaptativa + Memoria Volumétrica + L1 Pruning
- Cuantización guiada por varianza de activaciones.
- Memoria Volumétrica: Reemplazo de F.linear por Similitud Coseno en el lm_head (Tolerancia a fallos).
- L1 Pruning Condicional: Activado vía ENABLE_PRUNING=1 para comprimir aún más con Zlib.
- RUTAS FIJAS A SP1024 (10 Shards).
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS (FIJADOS A 1024)
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    
    enable_pruning = bool(int(os.environ.get("ENABLE_PRUNING", "0")))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "0")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    codebook_size = int(os.environ.get("CODEBOOK_SIZE", 256))
    calib_batches = int(os.environ.get("CALIB_BATCHES", 8))
    sdclip_k_base = float(os.environ.get("SDCLIP_K_BASE", 12.85))
    sdclip_k_min = float(os.environ.get("SDCLIP_K_MIN", 6.0))
    sdclip_k_max = float(os.environ.get("SDCLIP_K_MAX", 32.0))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]; lr = group["lr"]; momentum = group["momentum"]
            backend_steps = group["backend_steps"]; nesterov = group["nesterov"]
            if not params: continue
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# DATA LOADERS & EVAL
# -----------------------------
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith(" "): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            rs = bss * args.train_seq_len; re = bse * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): batch_loss = model(x, y).detach()
            btc = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * btc
            val_token_count += btc
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank; self.world_size = world_size; self.device = device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# CODEBOOK K-MEANS
# -----------------------------
def quantize_embedding_codebook(weight: Tensor, n_codes: int = 256):
    w = weight.float().cpu()
    vocab_size, dim = w.shape
    torch.manual_seed(42)
    perm = torch.randperm(vocab_size)[:n_codes]
    centers = w[perm].clone()
    labels = torch.zeros(vocab_size, dtype=torch.long)
    for _ in range(300):
        dists = torch.cdist(w, centers)
        new_labels = dists.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for k in range(n_codes):
            mask = new_labels == k
            if mask.any(): new_centers[k] = w[mask].mean(dim=0)
            else: new_centers[k] = centers[k]
        if (new_labels == labels).all(): break
        labels = new_labels; centers = new_centers
    indices = labels.numpy().astype(np.uint8)
    codebook = centers.numpy().astype(np.float32)
    mse = float(np.mean((w.numpy() - codebook[indices]) ** 2))
    return indices, codebook, mse

def dequantize_embedding_codebook(indices: np.ndarray, codebook: np.ndarray) -> Tensor:
    return torch.from_numpy(codebook[indices].astype(np.float32))

# -----------------------------
# CUANTIZACIÓN ADAPTATIVA (Díaz-Cano)
# -----------------------------
def collect_activation_stats(model, train_loader, device, n_batches=8, seq_len=1024, grad_accum_steps=4, train_batch_tokens=524288):
    activation_stats = {}
    hooks = []
    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0].detach().float()
            x_flat = x.reshape(-1, x.shape[-1])
            col_std = x_flat.std(dim=0)
            w = module.weight.float()
            row_importance = (w.abs() * col_std.unsqueeze(0)).mean(dim=1)
            if name not in activation_stats: activation_stats[name] = row_importance
            else: activation_stats[name] = activation_stats[name] + row_importance
        return hook
    for n, m in model.named_modules():
        if isinstance(m, CastedLinear) and m.weight.numel() > 65536: hooks.append(m.register_forward_hook(make_hook(n)))
    model.eval()
    with torch.inference_mode():
        for _ in range(n_batches):
            x, y = train_loader.next_batch(train_batch_tokens, seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): model(x, y)
    for h in hooks: h.remove()
    model.train()
    for k in activation_stats: activation_stats[k] = activation_stats[k] / n_batches
    return activation_stats

def quantize_float_tensor_adaptive(t, k_base=12.85, k_min=6.0, k_max=32.0, row_importance=None, bits=6):
    t32 = t.float()
    half_levels = (1 << (bits - 1)) - 1
    if t32.ndim == 2:
        if row_importance is not None and row_importance.numel() == t32.shape[0]:
            imp = row_importance.to(t32.device).float()
            imp_norm = imp / (imp.mean().clamp_min(1e-7))
            k_per_row = (k_base * imp_norm).clamp(k_min, k_max)
            std_dev = t32.std(dim=1).clamp_min(1e-7)
            clip_abs = (k_per_row * std_dev).unsqueeze(1)
        else:
            std_dev = t32.std(dim=1, keepdim=True).clamp_min(1e-7)
            clip_abs = k_base * std_dev
        clipped = torch.maximum(torch.minimum(t32, clip_abs), -clip_abs)
        scale = (clip_abs / float(half_levels)).clamp_min(1e-7)
        q = torch.clamp(torch.round(clipped / scale), -half_levels, half_levels).to(torch.int8).contiguous()
        return q, scale.to(torch.float16).contiguous()
    std_dev = float(t32.std().clamp_min(1e-7).item())
    clip_abs_val = k_base * std_dev
    scale = torch.tensor(clip_abs_val / float(half_levels))
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs_val, clip_abs_val) / scale), -half_levels, half_levels).to(torch.int8).contiguous()
    return q, scale

CONTROL_TENSOR_NAME_PATTERNS = tuple(pattern for pattern in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS", "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights").split(",") if pattern)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(pattern for pattern in os.environ.get("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS)).split(",") if pattern)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16

def tensor_nbytes(t: Tensor) -> int: return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_state_dict_adaptive(state_dict, activation_stats, codebook_size=256, use_codebook=True, k_base=12.85, k_min=6.0, k_max=32.0):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    codebook_data = {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors","num_nonfloat_tensors","baseline_tensor_bytes","payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel()); stats["num_tensors"] += 1; stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if name == "tok_emb.weight" and t.ndim == 2 and use_codebook:
            indices, codebook, mse = quantize_embedding_codebook(t, n_codes=codebook_size)
            codebook_data[name] = {"indices": indices, "codebook": codebook, "mse": mse}
            stats["payload_bytes"] += indices.nbytes + codebook.nbytes; stats["num_float_tensors"] += 1
            continue
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1; passthrough[name] = t; stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept; stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        row_importance = next((imp.cpu() for stat_name, imp in activation_stats.items() if stat_name in name or name in stat_name), None)
        q, s = quantize_float_tensor_adaptive(t, k_base=k_base, k_min=k_min, k_max=k_max, row_importance=row_importance)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q; scales[name] = s; dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "adaptive_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "codebook_data": codebook_data}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_adaptive(obj):
    out = {}
    qmeta = obj.get("qmeta", {}); passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {}); codebook_data = obj.get("codebook_data", {})
    for name, cb in codebook_data.items(): out[name] = dequantize_embedding_codebook(cb["indices"], cb["codebook"])
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name]); s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str): ot = ot.to(getattr(torch, orig_dtype)).contiguous()
        out[name] = ot
    return out

# -----------------------------
# TRANSFORMER MODULES (VOLUMETRIC)
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._seq_len_cached = 0; self._cos_cached = self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]; self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2
    return torch.cat((x[..., :h]*cos + x[..., h:]*sin, x[..., :h]*(-sin) + x[..., h:]*cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads; self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False); self.c_k = CastedLinear(dim, kv_dim, bias=False); self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype); q = apply_rotary_emb(q, cos, sin); k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False); self.proj = CastedLinear(hidden, dim, bias=False); self.proj._zero_init = True
    def forward(self, x): return self.proj(torch.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init); self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32)); self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x, x0):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init):
        super().__init__()
        self.tie_embeddings = tie_embeddings; self.tied_embed_init_std = tied_embed_init_std; self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2; self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init) for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        if not self.tie_embeddings and self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.tied_embed_init_std)
            self.lm_head._zero_init = False
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False): nn.init.zeros_(m.weight)

    def forward(self, input_ids, target_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        
        # ==================================================================
        # MEMORIA VOLUMÉTRICA (Esferas Difusas)
        # ==================================================================
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        
        w_head = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        
        x_norm = F.normalize(x, p=2, dim=-1)
        w_norm = F.normalize(w_head, p=2, dim=-1)
        
        cosine_sim = F.linear(x_norm, w_norm)
        volumetric_radius = 30.0 
        lp = cosine_sim * volumetric_radius
        
        logits = self.logit_softcap * torch.tanh(lp / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

# -----------------------------
# MAIN TRAINING LOOP
# -----------------------------
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0: raise ValueError(f"WORLD_SIZE must be positive")
    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size: raise ValueError(f"VOCAB_SIZE mismatch")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"Pruning L1 Activado: {args.enable_pruning} | Memoria Volumétrica Activada")

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)

    mlp_params = [p for name, p in base_model.named_parameters() if "mlp.fc" in name or "mlp.proj" in name]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups: g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wdms = args.warmdown_iters * step_ms
        rem = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem / max(wdms, 1e-9) if rem <= wdms else 1.0

    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_st = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws_idx in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if (ws_idx + 1) % 10 == 0 or ws_idx + 1 == args.warmup_steps: log0(f"warmup_step:{ws_idx+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for opt, st in zip(optimizers, init_opt_st): opt.load_state_dict(st)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations: log0(f"stopping_early: wallclock_cap step:{step}/{args.iterations}")
            break

        elapsed_frac = (training_time_ms + 1000.0*(time.perf_counter()-t0)) / max_wallclock_ms if max_wallclock_ms else step / args.iterations
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
            train_loss += loss.detach()
            
            total_loss = loss * grad_scale
            
            if args.enable_pruning and elapsed_frac > 0.50:
                l1_penalty = sum(p.abs().sum() for p in mlp_params)
                l1_lambda = 1e-5 * ((elapsed_frac - 0.50) / 0.50)
                total_loss = total_loss + (l1_lambda * l1_penalty * grad_scale)
            
            total_loss.backward()
            
        train_loss /= grad_accum_steps

        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1-frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups: g["momentum"] = muon_mom
        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model (fp): {model_bytes} bytes | code: {code_bytes} bytes")

    log0(f"Recolectando estadísticas de activación ({args.calib_batches} batches)...")
    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    t_calib = time.perf_counter()
    activation_stats = collect_activation_stats(
        base_model, calib_loader, device,
        n_batches=args.calib_batches, seq_len=args.train_seq_len,
        grad_accum_steps=grad_accum_steps, train_batch_tokens=args.train_batch_tokens,
    )
    log0(f"Calibración completada en {time.perf_counter()-t_calib:.1f}s — {len(activation_stats)} capas")

    use_codebook = args.tie_embeddings
    log0(f"Cuantización adaptativa: use_codebook={use_codebook} k_base={args.sdclip_k_base}")

    quant_obj, quant_stats = quantize_state_dict_adaptive(
        base_model.state_dict(), activation_stats=activation_stats,
        codebook_size=args.codebook_size, use_codebook=use_codebook,
        k_base=args.sdclip_k_base, k_min=args.sdclip_k_min, k_max=args.sdclip_k_max,
    )

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)

    if master_process:
        with open("final_model.adaptive_v8_1.ptz", "wb") as f: f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.adaptive_v8_1.ptz")
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["payload_bytes"], 1)
        log0(f"Serialized adaptive+zlib: {quant_file_bytes} bytes (ratio:{ratio:.2f}x) total:{quant_file_bytes + code_bytes} bytes")

    if distributed: dist.barrier()
    with open("final_model.adaptive_v8_1.ptz", "rb") as f: blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_adaptive(quant_state), strict=True)
    
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    
    torch.cuda.synchronize(); t_q = time.perf_counter()
    q_loss, q_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    
    log0(f"final_adaptive_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} eval_time:{1000*(time.perf_counter()-t_q):.0f}ms")
    log0(f"final_adaptive_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()