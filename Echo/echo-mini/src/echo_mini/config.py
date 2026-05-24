"""echo-mini 模型配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EchoMiniConfig:
    """Llama-style decoder-only transformer 超参。"""

    vocab_size: int = 16_384
    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 4
    d_ff: int = 1_376
    max_seq_len: int = 1_024
    dropout: float = 0.0
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
