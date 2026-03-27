"""
MPO-PicoGPT: Compressing PicoGPT.jl with Matrix Product Operators in PyTorch
=============================================================================

PicoGPT.jl architecture (from README):
  vocab_size  = 65     (character-level, Shakespeare)
  d_model     = 128
  n_heads     = 4
  n_layers    = 4
  seq_len     = 256
  d_ff        = 512    (= 4 × d_model)
  ~1 M dense parameters

This file walks through the MPO compression in six steps:

  Step 1 — MPO math:     TT-SVD decomposition of a weight matrix
  Step 2 — MPOLinear:    trainable nn.Module with MPO-parameterised weight
  Step 3 — Dense GPT:    PicoGPT in PyTorch (exact architecture match)
  Step 4 — MPO-GPT:      replace every nn.Linear with MPOLinear
  Step 5 — Analysis:     compression ratios and reconstruction errors
  Step 6 — Smoke test:   forward pass + greedy generation demo

References
----------
  Novikov et al. 2015 "Tensorizing Neural Networks" (TT-MPS for FC layers)
  Oseledets 2011 "Tensor-Train Decomposition" (TT-SVD algorithm)
  Gao et al. 2020 "Compressing Deep Neural Networks by Matrix Product Operators"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — MPO MATH
# ══════════════════════════════════════════════════════════════════════════════
#
# A weight matrix  W ∈ ℝ^{out × in}  can be interpreted as an operator
# that maps vectors in ℝ^{in} to ℝ^{out}.
#
# If we factorise  out = d_out[0]·d_out[1]·…·d_out[L-1]
#              and  in  = d_in[0] ·d_in[1] ·…·d_in[L-1],
# we can write W as a rank-2L tensor and decompose it as an MPO:
#
#   W_{(i₀…i_{L-1}),(j₀…j_{L-1})}
#     = Σ_{α₁…α_{L-1}}
#         A[0]_{i₀,j₀,α₁}  ·  A[1]_{α₁,i₁,j₁,α₂}  ·  …  ·  A[L-1]_{α_{L-1},i_{L-1},j_{L-1}}
#
# Each core  A[l]  has shape  (χ_{l-1}, d_out[l], d_in[l], χ_l)  with χ_0 = χ_L = 1.
# The bond dimension χ controls the approximation quality vs parameter count.
#
# We find the cores with TT-SVD (Oseledets 2011):
#   1. Reshape & interleave W into T of shape (d_out[0],d_in[0],...,d_out[L-1],d_in[L-1])
#   2. Left-to-right SVD sweep: unfold each site, SVD, truncate to χ, peel off one core.
#
# ─────────────────────────────────────────────────────────────────────────────

def mpo_decompose(
    W: torch.Tensor,          # shape (out, in)
    d_out: List[int],         # factorisation of out  (product = W.shape[0])
    d_in:  List[int],         # factorisation of in   (product = W.shape[1])
    max_bond: int,            # maximum bond dimension χ
) -> List[torch.Tensor]:
    """
    TT-SVD decomposition of weight matrix W into MPO cores.

    Returns
    -------
    cores : list of L tensors
        core[0]   shape (1,           d_out[0], d_in[0], χ₁)
        core[l]   shape (χ_l,         d_out[l], d_in[l], χ_{l+1})
        core[L-1] shape (χ_{L-1},     d_out[-1], d_in[-1], 1)
    """
    L = len(d_out)
    assert len(d_in) == L, "d_out and d_in must have the same length"
    assert math.prod(d_out) == W.shape[0], f"prod(d_out)={math.prod(d_out)} != W.shape[0]={W.shape[0]}"
    assert math.prod(d_in)  == W.shape[1], f"prod(d_in)={math.prod(d_in)} != W.shape[1]={W.shape[1]}"

    # ── 1. Reshape W → interleaved tensor ────────────────────────────────────
    # W: (out, in) → (d_out[0], …, d_out[L-1], d_in[0], …, d_in[L-1])
    T = W.detach().reshape(d_out + d_in)

    # Interleave: (d_out[0], d_in[0], d_out[1], d_in[1], …, d_out[L-1], d_in[L-1])
    perm = []
    for l in range(L):
        perm += [l, L + l]
    T = T.permute(perm).contiguous()
    # T.shape = (d_out[0], d_in[0], …, d_out[L-1], d_in[L-1])

    # ── 2. Left-to-right SVD sweep ────────────────────────────────────────────
    cores = []
    bond_left = 1

    # Prepend a trivial bond-1 dimension so all sites look uniform
    T_work = T.reshape(1, *T.shape)  # (1, d_out[0], d_in[0], …)

    for l in range(L - 1):
        d_o, d_i = d_out[l], d_in[l]

        # Unfold: (bond_left·d_out[l]·d_in[l],  rest)
        left_dim  = bond_left * d_o * d_i
        right_dim = math.prod(T_work.shape[3:])          # everything after site l
        M = T_work.reshape(left_dim, right_dim)          # 2-D matrix

        # SVD with truncation
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = min(max_bond, S.shape[0])
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        # Peel off core l: reshape U → (bond_left, d_out[l], d_in[l], r)
        cores.append(U.reshape(bond_left, d_o, d_i, r))

        # Absorb Σ into V†, continue with the remainder
        rest_shape = list(T_work.shape[3:])              # dims of remaining sites
        T_work = (torch.diag(S) @ Vh).reshape(r, *rest_shape)
        bond_left = r

    # Last core: everything that remains — shape (bond_left, d_out[-1], d_in[-1], 1)
    cores.append(T_work.reshape(bond_left, d_out[-1], d_in[-1], 1))

    return cores


def mpo_to_matrix(
    cores: List[torch.Tensor],
    d_out: List[int],
    d_in:  List[int],
) -> torch.Tensor:
    """
    Contract MPO cores → full weight matrix of shape (out, in).

    Contraction:
        Start with core[0] (1, d_out[0], d_in[0], χ), squeeze leading dim.
        For each subsequent core (χ, d_out[l], d_in[l], χ'), contract the
        bond index (last dim of running result with first dim of core).
        After L sites: permute (out₀,in₀,…,out_{L-1},in_{L-1})
                    → (out₀,…,out_{L-1}, in₀,…,in_{L-1})
        Then reshape → (out, in).
    """
    L = len(cores)

    # Start: squeeze trivial leading bond
    result = cores[0].squeeze(0)  # (d_out[0], d_in[0], χ₁)

    for l in range(1, L):
        core = cores[l]  # (χ_l, d_out[l], d_in[l], χ_{l+1})
        # result: (*accumulated_dims, χ_l)
        # Contract last index of result with first index of core
        result = torch.tensordot(result, core, dims=([-1], [0]))
        # result: (*accumulated_dims, d_out[l], d_in[l], χ_{l+1})

    # result shape: (d_out[0], d_in[0], …, d_out[L-1], d_in[L-1], 1)
    result = result.squeeze(-1)   # drop trailing χ=1

    # Permute: interleaved (out0,in0,…) → separated (out0,…,outL-1, in0,…,inL-1)
    # Current dim order: [out0, in0, out1, in1, …, out_{L-1}, in_{L-1}]
    perm = list(range(0, 2 * L, 2)) + list(range(1, 2 * L, 2))
    result = result.permute(perm).contiguous()

    return result.reshape(math.prod(d_out), math.prod(d_in))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MPOLinear: trainable layer with MPO-parameterised weight
# ══════════════════════════════════════════════════════════════════════════════
#
# Replaces  nn.Linear(in_features, out_features)  with an equivalent layer
# whose weight matrix is represented as an MPO.
#
# Parameter count comparison (bond dim χ, L sites):
#   dense:  out × in
#   MPO:    Σ_l  χ_{l-1} · d_out[l] · d_in[l] · χ_l   (where χ_0=χ_L=1)
#
# Example — (512, 128) with L=3, d_out=[8,8,8], d_in=[4,4,8], χ=8:
#   dense : 512 × 128 = 65 536
#   MPO   : 1·8·4·8 + 8·8·8·8 + 8·8·8·1 = 256 + 4096 + 512 = 4 864  → 13.5×
# ─────────────────────────────────────────────────────────────────────────────

class MPOLinear(nn.Module):
    """
    Linear layer  y = x W^T + b  where W is stored as an MPO.

    Parameters
    ----------
    d_out, d_in : list of ints
        Per-site physical dimensions. Must satisfy prod(d_out)=out_features,
        prod(d_in)=in_features.  Both lists must have the same length L.
    bond_dim : int
        Maximum bond dimension χ.  Larger → better approximation, more params.
    bias : bool
        If True, adds a learnable bias of shape (out_features,).
    """

    def __init__(self, d_out: List[int], d_in: List[int],
                 bond_dim: int, bias: bool = True):
        super().__init__()
        L = len(d_out)
        assert len(d_in) == L

        self.d_out       = d_out
        self.d_in        = d_in
        self.L           = L
        self.out_features = math.prod(d_out)
        self.in_features  = math.prod(d_in)
        self.bond_dim    = bond_dim

        # Bond dimensions:  1 — χ — χ — … — χ — 1
        bonds = [1] + [bond_dim] * (L - 1) + [1]

        # Initialise cores with small random values scaled by 1/√(d_o·d_i·χ)
        # so that the reconstructed W has entries ~O(1/√in_features)
        self.cores = nn.ParameterList([
            nn.Parameter(
                torch.randn(bonds[l], d_out[l], d_in[l], bonds[l + 1])
                * (self.in_features ** -0.25)          # heuristic init
                / (bond_dim ** (0.5 * (L - 1) / L))   # spread across sites
            )
            for l in range(L)
        ])

        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    # ── class method: build from a pretrained nn.Linear by compression ────────
    @classmethod
    def from_linear(cls, linear: nn.Linear,
                    d_out: List[int], d_in: List[int],
                    bond_dim: int) -> "MPOLinear":
        """
        Compress a pretrained nn.Linear into an MPOLinear.

        The weight matrix is decomposed via TT-SVD with rank truncation to
        bond_dim. A truncation error is printed for each layer.
        """
        layer = cls(d_out, d_in, bond_dim, bias=(linear.bias is not None))

        W = linear.weight.data   # (out, in)
        cores = mpo_decompose(W, d_out, d_in, bond_dim)

        # Load cores into parameters
        for param, core in zip(layer.cores, cores):
            param.data = core.clone()

        # Measure reconstruction error
        W_approx = mpo_to_matrix(cores, d_out, d_in)
        rel_err = (W - W_approx).norm() / W.norm()
        print(f"  MPO compression  shape={tuple(W.shape)}  χ={bond_dim}  "
              f"rel_err={rel_err:.4f}  "
              f"params {W.numel():,} → {layer.n_params():,}  "
              f"({layer.compression_ratio():.1f}×)")

        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    # ── forward: contract MPO → W, then apply linear ──────────────────────────
    def get_weight(self) -> torch.Tensor:
        """Contract all cores to produce the full weight matrix (out, in)."""
        return mpo_to_matrix(list(self.cores), self.d_out, self.d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.get_weight(), self.bias)

    # ── bookkeeping ────────────────────────────────────────────────────────────
    def n_params(self) -> int:
        total = sum(c.numel() for c in self.cores)
        if self.bias is not None:
            total += self.bias.numel()
        return total

    def compression_ratio(self) -> float:
        dense_params = self.out_features * self.in_features
        if self.bias is not None:
            dense_params += self.out_features
        return dense_params / self.n_params()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"L={self.L}, χ={self.bond_dim}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DENSE PICOGPT IN PYTORCH
# ══════════════════════════════════════════════════════════════════════════════
#
# Exact architecture match for PicoGPT.jl (from the README):
#   • Pre-norm residual (GPT-2 style)
#   • Sinusoidal positional encoding (no learnable PE)
#   • Scaled dot-product multi-head causal self-attention
#   • FFN with ReLU activation (not GeLU)
#   • Character-level tokenisation, vocab_size = 65
#
# Linear layers per transformer block:
#   W_q  (128, 128)        query projection
#   W_k  (128, 128)        key projection
#   W_v  (128, 128)        value projection
#   W_o  (128, 128)        attention output projection
#   W1   (512, 128)        FFN up-projection
#   W2   (128, 512)        FFN down-projection
#
# Plus:
#   lm_head (65, 128)      logit projection
#   embedding (65, 128)    token embedding (shared weights optional)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size : int   = 65
    d_model    : int   = 128
    n_heads    : int   = 4
    n_layers   : int   = 4
    seq_len    : int   = 256
    d_ff       : int   = 512     # = 4 × d_model
    dropout    : float = 0.0


def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding, shape (seq_len, d_model)."""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(0, d_model, 2, dtype=torch.float32)
    freq = 1.0 / (10000 ** (dim / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses four separate projections (W_q, W_k, W_v, W_o), each (d_model, d_model),
    matching PicoGPT.jl's explicit tensor layout rather than a fused QKV matrix.
    This makes per-matrix MPO compression straightforward.
    """

    def __init__(self, cfg: GPTConfig, linear_cls=nn.Linear):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        D = cfg.d_model

        # Four separate (D, D) projections — easy to swap with MPOLinear
        self.W_q = linear_cls(D, D, bias=False)
        self.W_k = linear_cls(D, D, bias=False)
        self.W_v = linear_cls(D, D, bias=False)
        self.W_o = linear_cls(D, D, bias=True)

        # Causal mask — not a parameter, just a buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool))
            .view(1, 1, cfg.seq_len, cfg.seq_len)
        )
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        # Project and split heads
        def split_heads(z):
            return z.view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)

        q = split_heads(self.W_q(x))
        k = split_heads(self.W_k(x))
        v = split_heads(self.W_v(x))

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, H, T, T)
        attn = attn.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        y = attn @ v                                       # (B, H, T, Dh)
        y = y.transpose(1, 2).reshape(B, T, D)            # (B, T, D)
        return self.W_o(y)


class FFN(nn.Module):
    """Position-wise feed-forward network with ReLU (matching PicoGPT.jl)."""

    def __init__(self, cfg: GPTConfig, linear_cls=nn.Linear):
        super().__init__()
        self.W1 = linear_cls(cfg.d_model, cfg.d_ff,    bias=True)
        self.W2 = linear_cls(cfg.d_ff,    cfg.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(F.relu(self.W1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: h ← h + Attn(LN(h)) + FFN(LN(h))."""

    def __init__(self, cfg: GPTConfig, linear_cls=nn.Linear):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, linear_cls)
        self.ffn  = FFN(cfg, linear_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PicoGPT(nn.Module):
    """
    Full PicoGPT model.

    Tensor shapes (batch-first, unlike PicoGPT.jl's column-major D×T×B):
        tokens   : (B, T)   integer token ids
        hidden   : (B, T, D)
        logits   : (B, T, V)
    """

    def __init__(self, cfg: GPTConfig, linear_cls=nn.Linear):
        super().__init__()
        self.cfg        = cfg
        self.tok_emb    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer("pos_enc", sinusoidal_pe(cfg.seq_len, cfg.d_model))
        self.blocks     = nn.ModuleList(
            [TransformerBlock(cfg, linear_cls) for _ in range(cfg.n_layers)]
        )
        self.ln_f       = nn.LayerNorm(cfg.d_model)
        self.lm_head    = linear_cls(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        """GPT-2-style weight initialisation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.seq_len

        x = self.tok_emb(idx) + self.pos_enc[:T]        # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)                                 # (B, T, D)
        logits = self.lm_head(x)                         # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive generation (greedy / temperature / top-k)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature    # (B, V)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MPO-GPT: replace every nn.Linear with MPOLinear
# ══════════════════════════════════════════════════════════════════════════════
#
# We choose factorisation shapes so that the physical indices have roughly
# equal size (balanced MPO), which maximises compression.
#
# PicoGPT linear shapes and their factorizations
# ───────────────────────────────────────────────
#  Layer        shape        d_out            d_in         sites L
#  W_q,k,v,o   (128, 128)   [8, 16]          [8, 16]       2
#  W1  (up)    (512, 128)   [8, 8, 8]        [4, 4, 8]     3
#  W2  (down)  (128, 512)   [4, 4, 8]        [8, 8, 8]     3
#  lm_head     (65, 128)    [5, 13]          [8, 16]        2
#                            (65=5×13)        (128=8×16)
#
# Parameter counts with χ = 8
# ───────────────────────────
#  (128,128) L=2 : 1·8·8·8 + 8·16·16·1 = 512 + 2048 = 2 560  (vs 16 384) → 6.4×
#  (512,128) L=3 : 1·8·4·8 + 8·8·8·8  + 8·8·8·1 = 256+4096+512 = 4 864   → 13.5×
#  (128,512) L=3 : 1·4·8·8 + 8·4·8·8  + 8·8·8·1 = 256+2048+512 = 2816    → ~18×
#  (65,128)  L=2 : 1·5·8·8 + 8·13·16·1 = 320 + 1664 = 1 984  (vs 8 320)  → 4.2×
#
# Per block (4 attn + 2 FF):
#   Dense : 4×16384 + 65536 + 65536 = 196 608
#   MPO   : 4×2560  + 4864 + 2816  = 17 920
#   Block compression ≈ 11×
# ─────────────────────────────────────────────────────────────────────────────

# Factorisation registry — keyed by (out_features, in_features)
FACTORIZATIONS = {
    (128, 128): {"d_out": [8,  16],   "d_in": [8,  16]},
    (512, 128): {"d_out": [8,  8, 8], "d_in": [4,  4, 8]},
    (128, 512): {"d_out": [4,  4, 8], "d_in": [8,  8, 8]},
    (65,  128): {"d_out": [5,  13],   "d_in": [8,  16]},
}


def make_mpo_linear(in_features: int, out_features: int,
                    bond_dim: int, bias: bool = True) -> MPOLinear:
    """
    Build a fresh (randomly initialised) MPOLinear for the given shape.
    Raises KeyError if the shape is not in the factorisation registry.
    """
    key = (out_features, in_features)
    if key not in FACTORIZATIONS:
        raise KeyError(
            f"No MPO factorisation defined for shape {key}. "
            f"Add an entry to FACTORIZATIONS."
        )
    f = FACTORIZATIONS[key]
    return MPOLinear(f["d_out"], f["d_in"], bond_dim, bias=bias)


def mpo_linear_cls(bond_dim: int):
    """
    Returns a drop-in replacement for nn.Linear that uses MPOLinear.
    Used as the `linear_cls` argument to PicoGPT / layer constructors.
    """
    def _cls(in_features: int, out_features: int, bias: bool = True):
        return make_mpo_linear(in_features, out_features, bond_dim, bias)
    return _cls


def MPO_PicoGPT(cfg: GPTConfig, bond_dim: int = 8) -> PicoGPT:
    """
    Build PicoGPT with all Linear layers replaced by MPOLinear.

    Usage:
        model = MPO_PicoGPT(GPTConfig(), bond_dim=8)
    """
    return PicoGPT(cfg, linear_cls=mpo_linear_cls(bond_dim))


def compress_pretrained(dense_model: PicoGPT,
                        bond_dim: int = 8) -> PicoGPT:
    """
    Create an MPO-PicoGPT initialised from a pretrained dense PicoGPT.

    Each nn.Linear weight is compressed via TT-SVD and loaded into the
    corresponding MPOLinear.  A reconstruction error is printed for each.

    Usage:
        dense = PicoGPT(GPTConfig())
        # ... train dense ...
        mpo   = compress_pretrained(dense, bond_dim=8)
        # ... fine-tune mpo ...
    """
    cfg = dense_model.cfg
    mpo_model = MPO_PicoGPT(cfg, bond_dim=bond_dim)

    def _get_matching_mpo(dense_linear: nn.Linear,
                          mpo_linear: MPOLinear) -> None:
        key = (dense_linear.out_features, dense_linear.in_features)
        f   = FACTORIZATIONS[key]
        cores = mpo_decompose(
            dense_linear.weight.data,
            f["d_out"], f["d_in"],
            bond_dim
        )
        W_approx = mpo_to_matrix(cores, f["d_out"], f["d_in"])
        rel_err  = (dense_linear.weight.data - W_approx).norm() / \
                    dense_linear.weight.data.norm()
        print(f"  {key}  χ={bond_dim}  rel_err={rel_err:.4f}  "
              f"params {dense_linear.weight.numel():,} → "
              f"{sum(c.numel() for c in cores):,}  "
              f"({dense_linear.weight.numel() / sum(c.numel() for c in cores):.1f}×)")
        for param, core in zip(mpo_linear.cores, cores):
            param.data = core.clone()
        if dense_linear.bias is not None and mpo_linear.bias is not None:
            mpo_linear.bias.data.copy_(dense_linear.bias.data)

    print("=== Compressing dense PicoGPT → MPO-PicoGPT ===")
    print(f"Bond dimension χ = {bond_dim}\n")

    # Walk through matching pairs (dense, mpo) for all Linear layers
    dense_linears = [(n, m) for n, m in dense_model.named_modules()
                     if isinstance(m, nn.Linear)]
    mpo_linears   = [(n, m) for n, m in mpo_model.named_modules()
                     if isinstance(m, MPOLinear)]

    assert len(dense_linears) == len(mpo_linears), \
        "Dense and MPO models have different numbers of linear layers"

    for (dn, dl), (mn, ml) in zip(dense_linears, mpo_linears):
        print(f"[{dn}]")
        _get_matching_mpo(dl, ml)

    return mpo_model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — COMPRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compression_report(dense_model: PicoGPT,
                       bond_dims: List[int] = (4, 8, 16, 32)) -> None:
    """
    Print a table of compression ratios and parameter counts for each χ.

    For each bond dimension, we also estimate the reconstruction error
    of each linear layer's weight matrix.
    """
    cfg = dense_model.cfg

    print("\n" + "=" * 72)
    print(f"{'MPO-PicoGPT Compression Report':^72}")
    print(f"{'Architecture: ' + str(cfg):^72}")
    print("=" * 72)

    dense_params = dense_model.n_params()
    print(f"\n  Dense PicoGPT parameters: {dense_params:,}\n")

    header = f"  {'χ':>4}  {'MPO params':>12}  {'Ratio':>8}  {'Max rel-err':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for chi in bond_dims:
        mpo_model = MPO_PicoGPT(cfg, bond_dim=chi)
        mpo_params = mpo_model.n_params()
        ratio      = dense_params / mpo_params

        # Measure worst-case reconstruction error across all layers
        max_err = 0.0
        for (dn, dl), (mn, ml) in zip(
            [(n, m) for n, m in dense_model.named_modules() if isinstance(m, nn.Linear)],
            [(n, m) for n, m in mpo_model.named_modules()   if isinstance(m, MPOLinear)],
        ):
            key = (dl.out_features, dl.in_features)
            f   = FACTORIZATIONS[key]
            cores   = mpo_decompose(dl.weight.data, f["d_out"], f["d_in"], chi)
            W_approx = mpo_to_matrix(cores, f["d_out"], f["d_in"])
            err = (dl.weight.data - W_approx).norm() / dl.weight.data.norm()
            max_err = max(max_err, err.item())

        print(f"  {chi:>4}  {mpo_params:>12,}  {ratio:>8.2f}×  {max_err:>12.4f}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SMOKE TEST: forward pass and toy generation
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 60)
    print("  PicoGPT MPO Compression — Smoke Test")
    print("=" * 60)

    cfg = GPTConfig()
    torch.manual_seed(42)

    # ── 6a. Build dense reference model ──────────────────────────────────────
    print("\n[Step 3] Building dense PicoGPT...")
    dense = PicoGPT(cfg)
    print(f"  Dense params: {dense.n_params():,}")

    # ── 6b. Forward pass check ───────────────────────────────────────────────
    print("\n[Step 3] Forward pass check...")
    B, T = 2, 32
    idx  = torch.randint(0, cfg.vocab_size, (B, T))
    tgt  = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = dense(idx, tgt)
    assert logits.shape == (B, T, cfg.vocab_size), f"Bad logit shape: {logits.shape}"
    print(f"  logits shape: {logits.shape}  loss: {loss.item():.4f}  ✓")

    # ── 6c. Compression analysis across bond dimensions ──────────────────────
    print("\n[Step 5] Compression report...")
    compression_report(dense, bond_dims=[4, 8, 16, 32, 64])

    # ── 6d. Build MPO-GPT (fresh init, χ=8) ─────────────────────────────────
    print("[Step 4] Building MPO-PicoGPT (fresh init, χ=8)...")
    mpo8 = MPO_PicoGPT(cfg, bond_dim=8)
    print(f"  MPO params: {mpo8.n_params():,}  "
          f"(ratio: {dense.n_params() / mpo8.n_params():.1f}×)")

    # ── 6e. MPO forward pass ─────────────────────────────────────────────────
    print("\n[Step 4] MPO forward pass check...")
    logits_mpo, loss_mpo = mpo8(idx, tgt)
    assert logits_mpo.shape == (B, T, cfg.vocab_size)
    print(f"  logits shape: {logits_mpo.shape}  loss: {loss_mpo.item():.4f}  ✓")

    # ── 6f. Compress from pretrained dense ───────────────────────────────────
    print("\n[Step 4] Compressing pretrained dense model to MPO (χ=8)...")
    mpo_from_dense = compress_pretrained(dense, bond_dim=8)

    # ── 6g. Generation comparison ─────────────────────────────────────────────
    print("\n[Step 6] Generation comparison (greedy, 30 tokens)...")
    prompt = torch.zeros(1, 1, dtype=torch.long)   # token 0 = first char

    with torch.no_grad():
        gen_dense = dense.generate(prompt.clone(), max_new_tokens=30,
                                   temperature=1.0, top_k=None)
        gen_mpo   = mpo_from_dense.generate(prompt.clone(), max_new_tokens=30,
                                             temperature=1.0, top_k=None)

    print(f"  Dense tokens: {gen_dense[0].tolist()}")
    print(f"  MPO   tokens: {gen_mpo[0].tolist()}")

    # Token-level agreement
    agree = (gen_dense == gen_mpo).float().mean().item()
    print(f"  Token agreement (compressed init): {agree:.1%}")

    # ── 6h. Logit comparison dense vs MPO (compressed from same weights) ──────
    with torch.no_grad():
        logits_d, _ = dense(idx)
        logits_m, _ = mpo_from_dense(idx)

    logit_err = (logits_d - logits_m).abs().mean().item()
    print(f"\n  Mean |logit_dense - logit_MPO| after compression: {logit_err:.5f}")
    print("  (This should be small relative to logit magnitudes.)")

    print("\n✓  All checks passed.")
    print("\n" + "=" * 60)
    print("  How to fine-tune the MPO model")
    print("=" * 60)
    print("""
  The MPO cores are ordinary nn.Parameters — you can fine-tune with
  any standard PyTorch optimizer:

      optimizer = torch.optim.AdamW(mpo_model.parameters(), lr=3e-4)

      for batch in dataloader:
          idx, targets = batch
          logits, loss = mpo_model(idx, targets)
          optimizer.zero_grad()
          loss.backward()          # gradients flow through mpo_to_matrix()
          optimizer.step()

  The gradient ∂L/∂A[l] is computed automatically by PyTorch through
  tensordot → permute → reshape, which are all differentiable ops.
  No custom backward needed.

  Tips for fine-tuning:
    • Start with a small learning rate (~1e-4) since the MPO init
      already approximates the dense weights.
    • Gradient clipping (max_norm=1.0) is recommended as in PicoGPT.jl.
    • For better quality, increase χ (diminishing returns past ~32
      for this 1M-param model).
    • 'get_weight()' is called every forward pass.  For inference
      efficiency, cache the weight matrices:
          W = layer.get_weight().detach()
      and use them directly.
""")


# ══════════════════════════════════════════════════════════════════════════════
# APPENDIX — MATHEMATICAL NOTES
# ══════════════════════════════════════════════════════════════════════════════
"""
Parameter counts
────────────────
Dense layer (out, in):
    N_dense = out × in  [+ out for bias]

MPO layer with L sites and bond dim χ:
    N_MPO = d_out[0]·d_in[0]·χ
          + Σ_{l=1}^{L-2}  χ·d_out[l]·d_in[l]·χ
          + χ·d_out[L-1]·d_in[L-1]
           [+ out for bias]

Compression ratio ≈ (d_out·d_in)^L  / ((L-2)·χ²·d_out_avg·d_in_avg + …)

TT-SVD complexity
─────────────────
The TT-SVD sweep performs L-1 SVDs.  The l-th SVD acts on a matrix of size
(χ_{l-1}·d_out[l]·d_in[l]) × (rest), so cost is O(χ·d²·n) overall.
For PicoGPT matrices this is fast on CPU (< 1ms per layer).

Gradient flow
─────────────
∂L/∂A[l]  =  ∂L/∂W  ·  (∂W/∂A[l])

∂W/∂A[l] is the Jacobian of the contraction w.r.t. site l, which equals
the product of all other cores (the "environment" of site l in DMRG
language).  PyTorch computes this automatically via autograd through the
tensordot chain.

For more efficient training one can alternate single-site updates (fixing
all but one core and solving the linear problem) — this is exactly the
single-site DMRG sweep, and corresponds to ALS (Alternating Least Squares)
for tensor completion.
"""

if __name__ == "__main__":
    smoke_test()
