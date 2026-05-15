"""ch07 练习 2：给 ch06 MiniGPT 加 KV cache。

两件事：
1. 数值一致：带 cache 的逐 token 生成结果，必须与无 cache 的整段 forward 完全一致
2. 性能加速：序列变长后，单步耗时几乎不增长（O(n) vs 无 cache 的 O(n²)）

实现策略：
- 不改 ch06 的原 Block / CausalSelfAttention（保持训练路径干净）
- 新建 BlockKV / CausalSelfAttentionKV，权重从原模型拷贝过来
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
CH06_DIR = REPO_ROOT / "Playground" / "ch06-transformer"
sys.path.insert(0, str(CH06_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from Echo.shared.device import get_device
from importlib import import_module
MiniGPT = import_module("03_model").MiniGPT
_ch06_block = import_module("02_block")
CausalSelfAttention = _ch06_block.CausalSelfAttention
FFN = _ch06_block.FFN


class CausalSelfAttentionKV(nn.Module):
    """带 KV cache 的因果自注意力。结构与 ch06 版完全相同，仅多 cache 逻辑。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        # cache：每层独立维护，shape (B, H, t_so_far, d_k)，t_so_far 随生成动态增长
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, n, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, n, d_k)
        k = k.view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.d_k).transpose(1, 2)

        if use_cache:
            if self.k_cache is not None:
                # 增量步：把新 K/V 拼到历史 cache 后面，得到完整历史 K/V
                # torch.cat(tensors, dim=2)：沿 seq 维（第 2 维）拼接，shape (B, H, t_prev+n, d_k)
                k = torch.cat([self.k_cache, k], dim=2)
                v = torch.cat([self.v_cache, v], dim=2)
            self.k_cache, self.v_cache = k, v
            # 增量步 query 只有"新输入"那部分，但 K/V 是全历史
            # 形状 q: (B, H, n_new, d_k), k/v: (B, H, t_total, d_k)
            # 当 n_new=1 时 attention 矩阵自动是 (1, t_total)，已经因果，无需 mask
            # 当 n_new>1 时（首次 prefill）才需要 causal mask
            is_causal = q.size(-2) > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        else:
            # 无 cache 路径，等价于 ch06 版
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, q.size(-2), d)
        return self.proj(out)


class BlockKV(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionKV(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPTKV(MiniGPT):
    """继承 MiniGPT，把 blocks 换成 BlockKV，添加 cache 推理路径。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 替换 blocks（结构同 ch06，多 use_cache 参数）
        d_model = self.token_emb.embedding_dim
        n_heads = self.blocks[0].attn.n_heads
        n_layers = len(self.blocks)
        # nn.ModuleList：与 list 区别在于会把内部 module 注册为子模块，
        # 让 .parameters() / .to(device) / state_dict 都能找到它们
        self.blocks = nn.ModuleList([BlockKV(d_model, n_heads) for _ in range(n_layers)])

    def reset_cache(self) -> None:
        for blk in self.blocks:
            blk.attn.reset_cache()

    def forward(self, ids: torch.Tensor, use_cache: bool = False, pos_offset: int = 0) -> torch.Tensor:
        """use_cache 模式下，pos_offset = 已生成 token 数（用于位置 embedding 索引）。"""
        B, n = ids.shape
        pos = torch.arange(pos_offset, pos_offset + n, device=ids.device)
        x = self.token_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x, use_cache=use_cache)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate_kv(self, ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """带 KV cache 的贪心生成。

        两阶段：
        - prefill：把 prompt 一次性 forward 进去，填充 cache
        - decode：每次只 forward 1 个 token，cache 增量
        """
        self.eval()
        self.reset_cache()
        # prefill：处理整个 prompt，cache 一次性建好
        logits = self(ids, use_cache=True, pos_offset=0)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)

        # decode：每步 1 个 token
        for _ in range(max_new_tokens - 1):
            cur_pos = ids.size(1) - 1  # 即将输入的 token 位置
            logits = self(next_id, use_cache=True, pos_offset=cur_pos)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(0)

    V, d, H, L, max_len = 64, 384, 6, 6, 1024
    # 用同一份初始化构造两个模型：一个无 cache 走 ch06 原版，一个带 cache
    model_no_cache = MiniGPT(V, d, H, L, max_len).to(device).eval()
    torch.manual_seed(0)
    model_kv = MiniGPTKV(V, d, H, L, max_len).to(device).eval()
    # load_state_dict(state_dict, strict=False)：strict=False 允许 state_dict 与模型存在
    # missing/unexpected key（这里两个模型的 blocks 子模块名相同但类不同，state_dict 会精确对齐）
    model_kv.load_state_dict(model_no_cache.state_dict(), strict=False)

    # ---- 验证 1：数值一致 ----
    prompt = torch.randint(0, V, (1, 8), device=device)
    n_new = 20

    # 无 cache 贪心
    ids_a = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = model_no_cache(ids_a[:, -max_len:])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids_a = torch.cat([ids_a, next_id], dim=1)

    # 有 cache 贪心
    ids_b = model_kv.generate_kv(prompt.clone(), max_new_tokens=n_new)

    diff = (ids_a != ids_b).sum().item()
    print(f"\n无 cache vs 有 cache 输出差异 token 数: {diff}（应为 0）")
    assert diff == 0, "KV cache 与无 cache 路径必须输出完全一致的 token 序列"

    # ---- 验证 2：性能加速 ----
    print("\n性能对比：生成 400 个 token（序列足够长 attention 才主导耗时）")
    prompt = torch.randint(0, V, (1, 4), device=device)
    n_new = 400

    # warm up（避免首次 CUDA kernel 编译开销混入计时）
    _ = model_kv.generate_kv(prompt.clone(), max_new_tokens=10)
    if device.type == "cuda":
        # CUDA kernel 调用是异步的，CPU 立刻返回。计时前必须 synchronize 等所有 kernel 跑完才准
        torch.cuda.synchronize()

    # 无 cache
    t0 = time.time()
    ids_a = prompt.clone()
    with torch.no_grad():
        for _ in range(n_new):
            logits = model_no_cache(ids_a[:, -max_len:])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids_a = torch.cat([ids_a, next_id], dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_no_cache = time.time() - t0

    # 有 cache
    t0 = time.time()
    _ = model_kv.generate_kv(prompt.clone(), max_new_tokens=n_new)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_with_cache = time.time() - t0

    print(f"  无 cache : {t_no_cache * 1000:.1f} ms ({t_no_cache * 1000 / n_new:.2f} ms/token)")
    print(f"  有 cache : {t_with_cache * 1000:.1f} ms ({t_with_cache * 1000 / n_new:.2f} ms/token)")
    speedup = t_no_cache / t_with_cache
    print(f"  加速比   : {speedup:.2f}x")
    # 序列足够长 + 模型足够大时，attention O(n²) 才会主导单步耗时
    # 阈值 1.5x 保守，3060 上一般能到 2-4x
    assert speedup > 1.5, f"KV cache 应有显著加速，实际 {speedup:.2f}x"

    print("\nPASS")


if __name__ == "__main__":
    main()
