# eval_loss 执行结果记录

## final

```bash
(base) sre@dev-office-ml-01:~/llm01-git/Echo/echo$ uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/final
Model: Qwen/Qwen2.5-1.5B
Adapter: checkpoints/sft/final
Val file: data/sft/val.jsonl
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 513.54it/s]
Val samples: 1000
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 730.03 examples/s]
Running eval...

Results:
  Adapter: checkpoints/sft/final
  Val samples: 1000
  Valid tokens: 869,158
  Avg loss: 1.4729
  Perplexity: 4.36
```

## 3000

```bash
(base) sre@dev-office-ml-01:~/llm01-git/Echo/echo$ uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/checkpoint-3000
Model: Qwen/Qwen2.5-1.5B
Adapter: checkpoints/sft/checkpoint-3000
Val file: data/sft/val.jsonl
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 508.24it/s]
Val samples: 1000
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 721.43 examples/s]
Running eval...

Results:
  Adapter: checkpoints/sft/checkpoint-3000
  Val samples: 1000
  Valid tokens: 869,158
  Avg loss: 1.4733
  Perplexity: 4.36
```

## 2000

```bash
(base) sre@dev-office-ml-01:~/llm01-git/Echo/echo$ uv run python scripts/eval_loss.py --adapter-dir checkpoints/sft/checkpoint-2000
Model: Qwen/Qwen2.5-1.5B
Adapter: checkpoints/sft/checkpoint-2000
Val file: data/sft/val.jsonl
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 528.62it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Val samples: 1000
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 723.14 examples/s]
Running eval...

Results:
  Adapter: checkpoints/sft/checkpoint-2000
  Val samples: 1000
  Valid tokens: 869,158
  Avg loss: 1.4181
  Perplexity: 4.13
```
