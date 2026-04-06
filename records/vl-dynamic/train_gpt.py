"""
train_gpt.py — Volumetric Logic (Ghost Spheres)
Training loop de OpenAI + bloque VL de Shakespeare que sabemos que funciona.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
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

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size   = int(os.environ.get("VAL_BATCH_SIZE",   524_288))
    val_loss_every   = int(os.environ.get("VAL_LOSS_EVERY",   1000))
    train_log_every  = int(os.environ.get("TRAIN_LOG_EVERY",  200))

    iterations        = int(os.environ.get("ITERATIONS",        20000))
    warmdown_iters    = int(os.environ.get("WARMDOWN_ITERS",    1200))
    warmup_steps      = int(os.environ.get("WARMUP_STEPS",      20))
    train_batch_tokens= int(os.environ.get("TRAIN_BATCH_TOKENS",524_288))
    train_seq_len     = int(os.environ.get("TRAIN_SEQ_LEN",     1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init      = float(os.environ.get("QK_GAIN_INIT",    1.5))

    vocab_size     = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers     = int(os.environ.get("NUM_LAYERS",    9))
    num_kv_heads   = int(os.environ.get("NUM_KV_HEADS",  4))
    model_dim      = int(os.environ.get("MODEL_DIM",     512))
    num_heads      = int(os.environ.get("NUM_HEADS",     8))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base      = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap  = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr       = float(os.environ.get("EMBED_LR",    0.6))
    head_lr        = float(os.environ.get("HEAD_LR",     0.008))
    tied_embed_lr  = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr      = float(os.environ.get("MATRIX_LR",  0.04))
    scalar_lr      = float(os.environ.get("SCALAR_LR",  0.04))
    muon_momentum  = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",  500))
    beta1          = float(os.environ.get("BETA1", 0.9))
    beta2          = float(os.environ.get("BETA2", 0.95))
    adam_eps       = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # VL — mismos ratios que Shakespeare ganador
    vl_k_max           = int(os.environ.get("VL_K_MAX",           1024))
    vl_k_init          = int(os.environ.get("VL_K_INIT",          256))
    vl_k_grow          = int(os.environ.get("VL_K_GROW",          64))
    vl_evo_interval    = int(os.environ.get("VL_EVO_INTERVAL",    500))
    vl_evo_cutoff_frac = float(os.environ.get("VL_EVO_CUTOFF_FRAC", 0.40))
    vl_prune_threshold = float(os.environ.get("VL_PRUNE_THRESHOLD", 0.001))
    vl_lr_mult         = float(os.environ.get("VL_LR_MULT",        4.0))
    vl_grad_clip       = float(os.environ.get("VL_GRAD_CLIP",      0.5))

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,log_radii,out_scale",
    ).split(",") if p
)

# -----------------------------
# MUON
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            total = sum(int(p.numel()) for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    if group["nesterov"]: g = g.add(buf, alpha=group["momentum"])
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(flat[curr: curr + p.numel()].view_as(p).to(p.dtype), alpha=-group["lr"])
                curr += p.numel()
        return loss

# -----------------------------
# DATA
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=offset)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(pattern)
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(pattern)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank; self.world_size = world_size
        self.device = device; self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# EVAL
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"): has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, bb_lut, hls_lut, ibt_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end   = (total_seqs * (rank + 1)) // world_size
    vls = torch.zeros((), device=device, dtype=torch.float64)
    vtc = torch.zeros((), device=device, dtype=torch.float64)
    vbc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            rs  = bss * args.train_seq_len
            re  = bse * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            btc = float(y.numel())
            vls += bl.to(torch.float64) * btc
            vtc += btc
            tb = bb_lut[y.reshape(-1)].to(dtype=torch.int16)
            tb += (hls_lut[y.reshape(-1)] & ~ibt_lut[x.reshape(-1)]).to(dtype=torch.int16)
            vbc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [vls, vtc, vbc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = vls / vtc
    bpt = val_loss.item() / math.log(2.0)
    tpb = vtc.item() / vbc.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)

# -----------------------------
# QUANTIZATION
# -----------------------------

INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_Q                 = 99.99984 / 100.0

def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name, t, pod):
    if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        pod[name] = str(t.dtype).removeprefix("torch.")
        return t.to(INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
        cl = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
        sc = (ca / 127.0).clamp_min(1.0 / 127.0)
        q  = torch.clamp(torch.round(cl / sc[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, sc.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    sc = torch.tensor(ca / 127.0 if ca > 0 else 1.0)
    q  = torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / sc), -127, 127).to(torch.int8).contiguous()
    return q, sc

def quantize_state_dict_int8(sd):
    Q, S, D, P, POD, QM = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                           "num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in sd.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel()); stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1; P[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t); continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, POD); P[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept); continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0: QM[name] = {"scheme": "per_row", "axis": 0}
        Q[name] = q; S[name] = s; D[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1",
           "quantized": Q, "scales": S, "dtypes": D, "passthrough": P}
    if QM:  obj["qmeta"] = QM
    if POD: obj["passthrough_orig_dtypes"] = POD
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    QM  = obj.get("qmeta", {})
    POD = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name]); s = obj["scales"][name]
        if QM.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().to("cpu").contiguous()
        od = POD.get(name)
        if isinstance(od, str): ot = ot.to(getattr(torch, od)).contiguous()
        out[name] = ot
    return out

# ==============================================================================
# VOLUMETRIC LOGIC — bloque Shakespeare que sabemos que funciona
# Float32 en parámetros geométricos, sin bfloat16 forzado
# ==============================================================================

class VolumetricLogic(nn.Module):
    """
    Ghost Spheres: zonas de influencia esféricas en espacio de activación.
    phi_k(h) = relu(1 - dist_coseno(h, c_k) / r_k)

    Clave: parámetros geométricos en float32, NO bfloat16.
    Centros inicializados desde activaciones reales (no random gaussiano).
    Grafo estático K_MAX compatible con torch.compile.
    """

    def __init__(self, dim: int, k_max: int, k_init: int):
        super().__init__()
        self.dim   = dim
        self.k_max = k_max

        # Float32 explícito — igual que en Shakespeare
        self.centers      = nn.Parameter(torch.zeros(k_max, dim, dtype=torch.float32))
        self.log_radii    = nn.Parameter(torch.zeros(k_max, dtype=torch.float32))
        self.push_vectors = nn.Parameter(torch.zeros(k_max, dim, dtype=torch.float32))
        self.out_scale    = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.register_buffer("k_active",        torch.tensor(k_init, dtype=torch.long))
        self.register_buffer("_activity_accum", torch.zeros(k_max, dtype=torch.float32))
        self.register_buffer("_activity_count", torch.tensor(0, dtype=torch.long))

        self.last_sparsity: float = 0.0
        self._init_weights(k_init)

    def _init_weights(self, k_init: int) -> None:
        with torch.no_grad():
            # Centros: gaussianos normalizados (se reemplazan con activaciones reales)
            c = F.normalize(torch.randn(k_init, self.dim), p=2, dim=-1)
            self.centers[:k_init].copy_(c)
            # Radio: softplus⁻¹(0.5) — igual que Shakespeare ganador
            init_r = math.log(math.exp(0.5) - 1.0)
            self.log_radii[:k_init].fill_(init_r)
            # Push vectors: Kaiming
            std = 2.0 / math.sqrt(self.dim)
            self.push_vectors[:k_init].normal_(0, std)

    def init_centers_from_activations(self, h: Tensor) -> None:
        """Inicializar centros desde activaciones reales. Llamar antes del warmup."""
        with torch.no_grad():
            k = int(self.k_active.item())
            h_flat = h.reshape(-1, self.dim).float()
            h_norm = F.normalize(h_flat, p=2, dim=-1)
            idx = torch.randint(0, h_norm.shape[0], (k,), device=h.device)
            self.centers.data[:k] = h_norm[idx]

    def forward(self, h: Tensor) -> Tensor:
        B, S, D = h.shape
        h_f = h.reshape(B * S, D).float()  # float32 para el bloque VL

        # Centros y radios — siempre K_MAX (grafo estático)
        c_norm = F.normalize(self.centers, p=2, dim=-1)
        radii  = F.softplus(self.log_radii) + 0.01

        # Distancia coseno + RBF
        h_norm = F.normalize(h_f, p=2, dim=-1)
        dist   = 1.0 - (h_norm @ c_norm.T)
        phi    = torch.relu(1.0 - dist / radii.unsqueeze(0))

        # Acumular actividad
        self._activity_accum.add_(phi.detach().mean(dim=0))
        self._activity_count.add_(1)
        self.last_sparsity = 0.0  # se mide fuera del grafo

        out = (phi @ self.push_vectors) * self.out_scale
        return out.reshape(B, S, D).to(h.dtype)

    @torch.no_grad()
    def prune(self, threshold: float = 0.001) -> int:
        count = int(self._activity_count.item())
        if count == 0: return 0
        k = int(self.k_active.item())
        avg_act = self._activity_accum[:k] / count
        keep    = avg_act >= threshold
        n_keep  = int(keep.sum().item())
        if n_keep == 0 or n_keep == k:
            self._reset_accum(); return 0
        idx = torch.where(keep)[0]
        self.centers.data[:n_keep]      = self.centers.data[idx]
        self.log_radii.data[:n_keep]    = self.log_radii.data[idx]
        self.push_vectors.data[:n_keep] = self.push_vectors.data[idx]
        self.centers.data[n_keep:k].zero_()
        self.log_radii.data[n_keep:k].zero_()
        self.push_vectors.data[n_keep:k].zero_()
        pruned = k - n_keep
        self.k_active.fill_(n_keep)
        self._reset_accum()
        return pruned

    @torch.no_grad()
    def grow(self, h_coords: Tensor, n_grow: int) -> int:
        k = int(self.k_active.item())
        n_real = min(n_grow, self.k_max - k)
        if n_real <= 0: return 0
        new_c = F.normalize(h_coords[:n_real].float(), p=2, dim=-1)
        specialist_r = math.log(math.exp(0.3) - 1.0)
        std = 2.0 / math.sqrt(self.dim)
        dev = self.centers.device
        self.centers.data[k:k+n_real]      = new_c.to(dev)
        self.log_radii.data[k:k+n_real]    = specialist_r
        self.push_vectors.data[k:k+n_real] = torch.randn(n_real, self.dim, device=dev) * std
        self.k_active.fill_(k + n_real)
        self._reset_accum()
        return n_real

    def _reset_accum(self):
        self._activity_accum.zero_()
        self._activity_count.zero_()

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len \
                or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None]
            self._sin_cached = freqs.sin()[None, None]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype), self._sin_cached.to(dtype)


def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2
    return torch.cat((x[..., :h] * cos + x[..., h:] * sin,
                      x[..., :h] * (-sin) + x[..., h:] * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 vl_k_max, vl_k_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = VolumetricLogic(dim, k_max=vl_k_max, k_init=vl_k_init)
        self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x   = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x   = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 vl_k_max, vl_k_init):
        super().__init__()
        self.tie_embeddings      = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap       = logit_softcap
        self.tok_emb  = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                  vl_k_max, vl_k_init)
            for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, 0.0, self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x  = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        lp = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(lp / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

# -----------------------------
# TRAINING
# -----------------------------

def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed  = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank         = int(os.environ.get("RANK", "0"))
    world_size   = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank   = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale   = 1.0 / grad_accum_steps

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Vocab mismatch: {args.vocab_size} vs {sp.vocab_size()}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb_lut, hls_lut, ibt_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    evo_cutoff_step = int(args.vl_evo_cutoff_frac * args.iterations)

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"VL config: k_max={args.vl_k_max} k_init={args.vl_k_init} "
         f"k_grow={args.vl_k_grow} evo_interval={args.vl_evo_interval} "
         f"evo_cutoff={evo_cutoff_step} Triton={'yes' if HAS_TRITON else 'no'}")

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        vl_k_max=args.vl_k_max, vl_k_init=args.vl_k_init,
    ).to(device).bfloat16()

    # VL params se mantienen en float32 — restaurar
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    # Asegurar que VL geométricos sean float32
    for block in base_model.blocks:
        block.mlp.centers.data      = block.mlp.centers.data.float()
        block.mlp.log_radii.data    = block.mlp.log_radii.data.float()
        block.mlp.push_vectors.data = block.mlp.push_vectors.data.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) \
        if distributed else compiled_model

    # Separar parámetros
    block_named = list(base_model.blocks.named_parameters())
    vl_names    = {'centers', 'log_radii', 'push_vectors'}
    ctrl_names  = set(CONTROL_TENSOR_NAME_PATTERNS)

    vl_params     = [p for n, p in block_named if any(x in n for x in vl_names)]
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2 and not any(x in n for x in vl_names)
                     and not any(x in n for x in ctrl_names)]
    scalar_params = [p for n, p in block_named
                     if (p.ndim < 2 or any(x in n for x in ctrl_names))
                     and not any(x in n for x in vl_names)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    vl_lr    = args.matrix_lr * args.vl_lr_mult

    opt_tok    = torch.optim.Adam([{"params": [base_model.tok_emb.weight],
                                    "lr": token_lr, "base_lr": token_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon   = Muon(matrix_params, lr=args.matrix_lr,
                      momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params,
                                    "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_vl     = torch.optim.Adam([{"params": vl_params, "lr": vl_lr, "base_lr": vl_lr}],
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps,
                                  fused=True, weight_decay=0.0)

    optimizers = [opt_tok, opt_muon, opt_scalar, opt_vl]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam([{"params": [base_model.lm_head.weight],
                                      "lr": args.head_lr, "base_lr": args.head_lr}],
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ==================================================================
    # INICIALIZAR CENTROS VL DESDE ACTIVACIONES REALES
    # Clave: en 512d los vectores aleatorios tienen dist_coseno ≈ 0.5
    # → phi = 0 siempre. Los centros deben estar donde están los datos.
    # ==================================================================
    log0("Inicializando centros VL desde activaciones reales...")
    with torch.no_grad():
        x_init, _ = train_loader.next_batch(
            args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        # Obtener activaciones después del embedding (donde opera el bloque VL)
        emb = base_model.tok_emb(x_init[:4, :256].to(device))
        emb = F.rms_norm(emb, (emb.size(-1),))  # mismo preproceso que en forward
        # Pasar por la primera capa de atención para obtener h real
        h_real = emb + base_model.blocks[0].attn_scale.to(emb.dtype)[None, None, :] * \
                 base_model.blocks[0].attn(base_model.blocks[0].attn_norm(emb))
        h_flat = h_real.reshape(-1, args.model_dim).float()
        h_norm = F.normalize(h_flat, p=2, dim=-1)
        for block in base_model.blocks:
            k = int(block.mlp.k_active.item())
            idx = torch.randint(0, h_norm.shape[0], (k,))
            block.mlp.centers.data[:k] = h_norm[idx]
        del x_init, emb, h_real, h_flat, h_norm
        torch.cuda.empty_cache()
    log0("Centros VL inicializados desde activaciones reales ✓")

    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds \
        if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wdms    = args.warmdown_iters * step_ms
        rem     = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem / max(wdms, 1e-9) if rem <= wdms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        init_state  = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_st = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws_idx in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if (ws_idx + 1) % 10 == 0 or ws_idx + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws_idx+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for opt, st in zip(optimizers, init_opt_st): opt.load_state_dict(st)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Loop principal
    training_time_ms = 0.0
    stop_after_step  = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_h_norm = None  # para mitosis: coordenadas de alta fricción

    step = 0
    while True:
        last_step = step == args.iterations or \
                    (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device,
                                         grad_accum_steps, val_tokens, bb_lut, hls_lut, ibt_lut)
            avg_sp = 0.0  # simplificado por ahora
            avg_k  = sum(int(b.mlp.k_active.item()) for b in base_model.blocks) / len(base_model.blocks)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"vl_sparsity:{avg_sp:.1%} vl_k_avg:{avg_k:.0f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1-frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = muon_mom

        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        # Clip extra agresivo para VL
        torch.nn.utils.clip_grad_norm_(vl_params, args.vl_grad_clip)
        for opt in optimizers: opt.step()
        zero_grad_all()

        # Mitosis
        if step > 0 and step % args.vl_evo_interval == 0 and step < evo_cutoff_step:
            total_pruned = total_grown = 0
            for block in base_model.blocks:
                total_pruned += block.mlp.prune(args.vl_prune_threshold)
                # Usar centros actuales normalizados como coords de crecimiento
                k = int(block.mlp.k_active.item())
                h_coords = F.normalize(
                    torch.randn(args.vl_k_grow, args.model_dim, device=device), p=2, dim=-1)
                total_grown += block.mlp.grow(h_coords, args.vl_k_grow)
            avg_k = sum(int(b.mlp.k_active.item()) for b in base_model.blocks) / len(base_model.blocks)
            log0(f"  [MITOSIS step {step}] pruned={total_pruned} grown={total_grown} k_avg={avg_k:.0f}")

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"model:{os.path.getsize('final_model.pt')} bytes | code:{code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    buf = io.BytesIO(); torch.save(quant_obj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f: f.write(blob)
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"int8+zlib:{os.path.getsize('final_model.int8.ptz')} bytes ratio:{ratio:.2f}x "
             f"total:{os.path.getsize('final_model.int8.ptz')+len(code.encode())}")

    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: blob_disk = f.read()
    state = torch.load(io.BytesIO(zlib.decompress(blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
    torch.cuda.synchronize()
    t_q = time.perf_counter()
    q_loss, q_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, bb_lut, hls_lut, ibt_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_time:{1000*(time.perf_counter()-t_q):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()