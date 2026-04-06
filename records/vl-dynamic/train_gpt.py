"""
train_gpt.py — Volumetric Logic Dynamic (VL) con kernel Triton Flash-Volumetric
Reemplaza el bloque MLP por Ghost Spheres con kernel fused memory-efficient.

KERNEL TRITON:
- Fusiona dist_coseno + RBF + push en un solo paso SRAM-bound
- atomic_add integrado para activity accumulator (sin pasada extra)
- Early exit por bloque cuando max(phi)==0 (explota sparsidad real)
- Tokens se cargan una sola vez en registros, esferas pasan en bloques
- Compatible con torch.compile fullgraph=True (grafo 100% estático)
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

# Triton opcional — fallback a PyTorch si no está disponible
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
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

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
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
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
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # VL params
    vl_k_max           = int(os.environ.get("VL_K_MAX",           512))
    vl_k_init          = int(os.environ.get("VL_K_INIT",          128))
    vl_k_grow          = int(os.environ.get("VL_K_GROW",          64))
    vl_evo_interval    = int(os.environ.get("VL_EVO_INTERVAL",    500))
    vl_evo_cutoff_frac = float(os.environ.get("VL_EVO_CUTOFF_FRAC", 0.40))
    vl_prune_threshold = float(os.environ.get("VL_PRUNE_THRESHOLD", 0.001))
    vl_init_radius     = float(os.environ.get("VL_INIT_RADIUS",    0.5))
    vl_lr_mult         = float(os.environ.get("VL_LR_MULT",        4.0))

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,log_radii,out_scale",
    ).split(",") if p
)

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# TOKENIZER / EVAL / DATA
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No train files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
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
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# QUANTIZATION
# -----------------------------

INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0

def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                           "num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1",
           "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# ==============================================================================
# KERNEL TRITON: FLASH-VOLUMETRIC
# ==============================================================================
# Fusiona: normalize(h) → dist_coseno → RBF → push → atomic_add(activity)
# Todo en SRAM. Los tokens se cargan UNA VEZ. Las esferas pasan en bloques.
# Early exit si max(phi)==0 para un bloque → explota sparsidad real.
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def flash_vl_fwd_kernel(
        h_ptr,        # (M, D) tokens en bfloat16
        c_ptr,        # (K, D) centros normalizados float32
        r_ptr,        # (K,)   radios softplus float32
        p_ptr,        # (K, D) push vectors bfloat16
        out_ptr,      # (M, D) output float32
        act_ptr,      # (K,)   activity accumulator float32
        M, K, D,
        stride_hm, stride_hd,
        stride_ck, stride_cd,
        stride_pk, stride_pd,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < M

        # Cargar bloque de tokens h → SRAM (una sola vez)
        h = tl.load(h_ptr + offs_m[:, None] * stride_hm + offs_d[None, :] * stride_hd,
                    mask=mask_m[:, None], other=0.0).to(tl.float32)

        # Normalizar h en registros (L2)
        h_sq = tl.sum(h * h, axis=1)
        h_norm_scale = 1.0 / tl.sqrt(h_sq + 1e-8)
        h = h * h_norm_scale[:, None]

        # Acumulador de salida en registros
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Iterar sobre esferas en bloques BLOCK_K
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Cargar centros (ya normalizados)
            c = tl.load(c_ptr + offs_k[:, None] * stride_ck + offs_d[None, :] * stride_cd,
                        mask=mask_k[:, None], other=0.0).to(tl.float32)

            # Distancia coseno: dot = h @ c.T → (BLOCK_M, BLOCK_K)
            dot = tl.dot(h, tl.trans(c))
            dist = 1.0 - dot

            # Radio
            r = tl.load(r_ptr + offs_k, mask=mask_k, other=1.0)

            # phi = relu(1 - dist/r)
            phi = tl.maximum(0.0, 1.0 - dist / r[None, :])

            # Early exit: si max(phi) == 0 en este bloque, ningún token activó
            # ninguna esfera → saltear push_vectors completamente
            phi_max = tl.max(phi)
            if phi_max > 0.0:
                # Acumular activity via atomic_add (fusionado, sin pasada extra)
                phi_sum = tl.sum(phi, axis=0)
                tl.atomic_add(act_ptr + offs_k, phi_sum, mask=mask_k)

                # Cargar push_vectors y acumular influencia
                pv = tl.load(p_ptr + offs_k[:, None] * stride_pk + offs_d[None, :] * stride_pd,
                             mask=mask_k[:, None], other=0.0).to(tl.float32)
                acc += tl.dot(phi.to(tl.bfloat16), pv.to(tl.bfloat16)).to(tl.float32)

        # Escribir output a VRAM (una sola vez)
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc, mask=mask_m[:, None])


    def flash_vl_forward(h: Tensor, centers: Tensor, radii: Tensor,
                         push_vectors: Tensor, activity_accum: Tensor) -> Tensor:
        """
        Wrapper del kernel Triton Flash-Volumetric.
        h:              (M, D) tokens (cualquier dtype)
        centers:        (K, D) centros normalizados
        radii:          (K,)   radios softplus
        push_vectors:   (K, D) vectores de empuje
        activity_accum: (K,)   acumulador de actividad (modificado in-place)
        returns:        (M, D) output float32
        """
        M, D = h.shape
        K    = centers.shape[0]
        dev  = h.device

        # Garantizar contiguidad y dtype correcto
        h_c  = h.to(torch.bfloat16).contiguous()
        c_c  = centers.to(torch.float32).contiguous()
        r_c  = radii.to(torch.float32).contiguous()
        pv_c = push_vectors.to(torch.bfloat16).contiguous()
        act_c = activity_accum.to(torch.float32).contiguous()

        out = torch.zeros(M, D, device=dev, dtype=torch.float32)

        # BLOCK_D debe ser potencia de 2 y <= D
        BLOCK_D = min(triton.next_power_of_2(D), D)
        BLOCK_M = 128
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M),)

        flash_vl_fwd_kernel[grid](
            h_c, c_c, r_c, pv_c, out, act_c,
            M, K, D,
            h_c.stride(0),  h_c.stride(1),
            c_c.stride(0),  c_c.stride(1),
            pv_c.stride(0), pv_c.stride(1),
            out.stride(0),  out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            num_warps=8, num_stages=3,
        )

        # Copiar activity de vuelta (atomic_add escribió en act_c)
        activity_accum.copy_(act_c)
        return out

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len \
                or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

# ==============================================================================
# VOLUMETRIC LOGIC CON KERNEL TRITON
# ==============================================================================

class VolumetricLogic(nn.Module):
    """
    Ghost Spheres con kernel Flash-Volumetric.
    Grafo estático (K_MAX fijo) compatible con torch.compile fullgraph.
    El kernel Triton fusiona dist+RBF+push+activity en una sola pasada SRAM.
    """

    def __init__(self, dim: int, k_max: int, k_init: int, init_radius: float = 0.5):
        super().__init__()
        self.dim   = dim
        self.k_max = k_max

        # Parámetros geométricos — shape fija K_MAX
        self.centers      = nn.Parameter(torch.zeros(k_max, dim))
        self.log_radii    = nn.Parameter(torch.zeros(k_max))
        self.push_vectors = nn.Parameter(torch.zeros(k_max, dim))
        self.out_scale    = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        # Puntero activo y acumuladores (buffers, no parámetros)
        self.register_buffer("k_active",         torch.tensor(k_init, dtype=torch.long))
        self.register_buffer("_activity_accum",  torch.zeros(k_max))
        self.register_buffer("_activity_count",  torch.tensor(0, dtype=torch.long))

        self.last_sparsity: float = 0.0
        self._init_weights(k_init, init_radius)

    def _init_weights(self, k_init: int, init_radius: float) -> None:
        with torch.no_grad():
            c = F.normalize(torch.randn(k_init, self.dim), p=2, dim=-1)
            self.centers[:k_init].copy_(c)
            init_r = math.log(math.exp(init_radius) - 1.0)
            self.log_radii[:k_init].fill_(init_r)
            std = 2.0 / math.sqrt(self.dim)
            self.push_vectors[:k_init].normal_(0, std)

    def forward(self, h: Tensor) -> Tensor:
        B, S, D = h.shape
        h_f = h.reshape(B * S, D)

        # Centros normalizados y radios (grafo estático: K_MAX siempre)
        c_norm = F.normalize(self.centers, p=2, dim=-1)
        radii  = F.softplus(self.log_radii) + 0.01

        if HAS_TRITON and h.is_cuda:
            # Kernel Triton: fusiona todo en SRAM, atomic_add activity integrado
            out = flash_vl_forward(
                h_f, c_norm, radii,
                self.push_vectors, self._activity_accum
            )
            self._activity_count.add_(1)
        else:
            # Fallback PyTorch puro
            h_norm = F.normalize(h_f, p=2, dim=-1)
            dist   = 1.0 - (h_norm @ c_norm.T)
            phi    = torch.relu(1.0 - dist / radii.unsqueeze(0))
            self._activity_accum.add_(phi.detach().mean(dim=0))
            self._activity_count.add_(1)
            out = (phi @ self.push_vectors).to(torch.float32)

        return (out.reshape(B, S, D) * self.out_scale.to(dtype=out.dtype))

    # ------------------------------------------------------------------
    # MITOSIS (fuera del grafo compilado)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune(self, threshold: float = 0.001) -> int:
        count = int(self._activity_count.item())
        if count == 0:
            return 0
        k = int(self.k_active.item())
        avg_act = self._activity_accum[:k] / count
        keep    = avg_act >= threshold
        n_keep  = int(keep.sum().item())
        if n_keep == 0 or n_keep == k:
            self._reset_accum()
            return 0
        keep_idx = torch.where(keep)[0]
        self.centers.data[:n_keep]      = self.centers.data[keep_idx]
        self.log_radii.data[:n_keep]    = self.log_radii.data[keep_idx]
        self.push_vectors.data[:n_keep] = self.push_vectors.data[keep_idx]
        self.centers.data[n_keep:k].zero_()
        self.log_radii.data[n_keep:k].zero_()
        self.push_vectors.data[n_keep:k].zero_()
        pruned = k - n_keep
        self.k_active.fill_(n_keep)
        self._reset_accum()
        return pruned

    @torch.no_grad()
    def grow(self, n_grow: int) -> int:
        k     = int(self.k_active.item())
        space = self.k_max - k
        n_real = min(n_grow, space)
        if n_real <= 0:
            return 0
        specialist_r = math.log(math.exp(0.3) - 1.0)
        std = 2.0 / math.sqrt(self.dim)
        dev = self.centers.device
        new_c = F.normalize(torch.randn(n_real, self.dim, device=dev), p=2, dim=-1)
        self.centers.data[k:k+n_real]      = new_c
        self.log_radii.data[k:k+n_real]    = specialist_r
        self.push_vectors.data[k:k+n_real] = torch.randn(n_real, self.dim, device=dev) * std
        self.k_active.fill_(k + n_real)
        self._reset_accum()
        return n_real

    def _reset_accum(self):
        self._activity_accum.zero_()
        self._activity_count.zero_()

# -----------------------------
# TRANSFORMER BLOCK
# -----------------------------

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 vl_k_max, vl_k_init, vl_init_radius):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = VolumetricLogic(dim, k_max=vl_k_max, k_init=vl_k_init,
                                    init_radius=vl_init_radius)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x   = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x   = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

# -----------------------------
# GPT
# -----------------------------

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 vl_k_max, vl_k_init, vl_init_radius):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                  vl_k_max, vl_k_init, vl_init_radius)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x  = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
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
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    evo_cutoff_step = int(args.vl_evo_cutoff_frac * args.iterations)

    log0(f"VL Flash-Volumetric kernel: {'Triton' if HAS_TRITON else 'PyTorch fallback'}")
    log0(f"VL config: k_max={args.vl_k_max} k_init={args.vl_k_init} "
         f"k_grow={args.vl_k_grow} radius={args.vl_init_radius} "
         f"evo_interval={args.vl_evo_interval} evo_cutoff={evo_cutoff_step}")
    log0(f"train_loader: {dataset_dir.name} shards:{actual_train_files}")

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        vl_k_max=args.vl_k_max, vl_k_init=args.vl_k_init,
        vl_init_radius=args.vl_init_radius,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    # torch.compile — VolumetricLogic.forward es fullgraph compatible
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) \
        if distributed else compiled_model

    # Separar parámetros
    block_named = list(base_model.blocks.named_parameters())
    vl_names    = {'centers', 'log_radii', 'push_vectors'}
    ctrl_names  = set(CONTROL_TENSOR_NAME_PATTERNS)

    vl_params = [p for n, p in block_named if any(x in n for x in vl_names)]
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2
                     and not any(x in n for x in vl_names)
                     and not any(x in n for x in ctrl_names)]
    scalar_params = [p for n, p in block_named
                     if (p.ndim < 2 or any(x in n for x in ctrl_names))
                     and not any(x in n for x in vl_names)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    vl_lr    = args.matrix_lr * args.vl_lr_mult

    optimizer_tok    = torch.optim.Adam([{"params": [base_model.tok_emb.weight],
                                          "lr": token_lr, "base_lr": token_lr}],
                                        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon   = Muon(matrix_params, lr=args.matrix_lr,
                            momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups: g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params": scalar_params,
                                          "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_vl     = torch.optim.Adam([{"params": vl_params,
                                          "lr": vl_lr, "base_lr": vl_lr}],
                                        betas=(args.beta1, args.beta2), eps=args.adam_eps,
                                        fused=True, weight_decay=0.0)

    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_vl]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam([{"params": [base_model.lm_head.weight],
                                      "lr": args.head_lr, "base_lr": args.head_lr}],
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        init_state  = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_st = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for opt, st in zip(optimizers, init_opt_st): opt.load_state_dict(st)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main loop
    training_time_ms = 0.0
    stop_after_step  = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or \
                    (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            avg_sp = sum(b.mlp.last_sparsity for b in base_model.blocks) / len(base_model.blocks)
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
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups: g["momentum"] = muon_mom

        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

        # Mitosis (fuera del grafo compilado)
        if step > 0 and step % args.vl_evo_interval == 0 and step < evo_cutoff_step:
            total_pruned = total_grown = 0
            for block in base_model.blocks:
                total_pruned += block.mlp.prune(args.vl_prune_threshold)
                total_grown  += block.mlp.grow(args.vl_k_grow)
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

    # Serialización
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"model size: {os.path.getsize('final_model.pt')} bytes | code: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f: f.write(blob)
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"int8+zlib: {os.path.getsize('final_model.int8.ptz')} bytes "
             f"(ratio:{ratio:.2f}x) total:{os.path.getsize('final_model.int8.ptz')+len(code.encode())}")

    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: blob_disk = f.read()
    state = torch.load(io.BytesIO(zlib.decompress(blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
    torch.cuda.synchronize()
    t_q = time.perf_counter()
    q_loss, q_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} "
         f"eval_time:{1000*(time.perf_counter()-t_q):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()