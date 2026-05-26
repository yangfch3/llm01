"""echo-mini 模型定义：Llama-style Decoder-only Transformer。

组件：RMSNorm, RoPE, GQA, SwiGLU FFN。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EchoMiniConfig


# ============================================================
# RMSNorm
# ============================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ============================================================
# Rotary Position Embedding (RoPE)
# ============================================================


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """预计算 RoPE 频率矩阵，返回 complex64 形状 (max_seq_len, dim//2)。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """对 x 施加 RoPE。x: (batch, seq_len, n_heads, head_dim)。"""
    # 把 head_dim 拆成 complex 对
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


# ============================================================
# Grouped Query Attention (GQA)
# ============================================================


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) + KV cache."""

    def __init__(self, cfg: EchoMiniConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # GQA repeat factor

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # KV cache (populated during inference)
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    def reset_cache(self) -> None:
        """清空 KV cache。"""
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # RoPE
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # KV cache: append and use full history
        if use_cache:
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k = self.cache_k
            v = self.cache_v

        kv_len = k.shape[1]

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                bsz, kv_len, self.n_heads, self.head_dim
            )
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                bsz, kv_len, self.n_heads, self.head_dim
            )

        # (bsz, n_heads, seq_len/kv_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.matmul(attn, v)  # (bsz, n_heads, seq_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


# ============================================================
# SwiGLU FFN
# ============================================================


class FeedForward(nn.Module):
    """SwiGLU FFN: gate + up → SiLU gate × up → down。"""

    def __init__(self, cfg: EchoMiniConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ============================================================
# Transformer Block
# ============================================================


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer block: RMSNorm → Attn → residual → RMSNorm → FFN → residual。"""

    def __init__(self, cfg: EchoMiniConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None, use_cache: bool = False
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs, mask, use_cache=use_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ============================================================
# EchoMini Model
# ============================================================


class EchoMini(nn.Module):
    """echo-mini: Llama-style causal language model (~60M params)。"""

    def __init__(self, cfg: EchoMiniConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: lm_head shares embedding weights
        self.lm_head.weight = self.tok_emb.weight

        # RoPE freqs buffer (not a parameter)
        freqs = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """SPEC §2.2 权重初始化。"""
        std = 0.02
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)

        for name, p in self.named_parameters():
            if p.dim() == 1:
                # RMSNorm weights → already ones, bias → zeros (none here)
                continue
            if "tok_emb" in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "wo" in name or "w_down" in name:
                # Output projections in attn / ffn get smaller std
                nn.init.normal_(p, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    def reset_cache(self) -> None:
        """清空所有层的 KV cache。"""
        for layer in self.layers:
            layer.attn.reset_cache()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: (batch, seq_len) token ids
            targets: (batch, seq_len) target ids for loss computation
            use_cache: 推理时启用 KV cache
            start_pos: KV cache 模式下当前 token 的位置偏移

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if targets provided, else None
        """
        bsz, seq_len = input_ids.shape
        assert start_pos + seq_len <= self.cfg.max_seq_len, (
            f"position {start_pos + seq_len} > max {self.cfg.max_seq_len}"
        )

        x = self.tok_emb(input_ids)
        freqs = self.rope_freqs[start_pos : start_pos + seq_len]

        # Causal mask
        if use_cache and start_pos > 0:
            # 增量推理：当前 token 可以看到所有已缓存位置，不需要 mask
            mask = None
        else:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, freqs, mask, use_cache=use_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss
