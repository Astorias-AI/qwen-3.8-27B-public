# Qwen3.8-27B NVFP4 on RTX 5090

Measured SGLang recipes for [Qwen3.8-27B NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) with the [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) draft model. Two hardware profiles, one winning stage each: `dflash2_hicache`.

Both weight repos are Apache-2.0. This repository is the serving recipe (pins and launch scripts), released under Apache-2.0. It does not contain model weights.

Runtime image: `lmsysorg/sglang:dev-qwen38-27b-dflash2` (CUDA 13, host driver 570 or newer, 580+ preferred).

## Profiles

| | 1× RTX 5090 | 2× RTX 5090 |
|---|---|---|
| Script | `serve/1x5090/` | `serve/2x5090/` |
| Tensor parallel | 1 | 2 |
| Concurrency | 4 | 8 |
| Mamba cache | 12 | 96 |
| Draft tokens | 4 | 6 |
| CUDA graph batch | 4 / 4 | 16 / 16 |
| HiCache ratio | 2 | 1 |
| `mem_fraction_static` | 0.90 | 0.86 |
| `max_total_tokens` | 65536 | 262144 |
| Context that actually allocated | 28,192 | 262,144 |
| Host RAM to look for | 64 GB | 128 GB |

On one 32 GB card the 65,536 pin does not bind. After weights, the draft model, and Mamba state, the live KV pool is 28,192 tokens. Two cards shard the weights and hold the model's full 262,144-token window.

## Measured output rate

On-box engine bench. Each profile has its own output length, so read each row inside its own table.

**1× T4B** (17 Sep 2026). Load steps are 2,048 in / 256 out.

| Workload | Output tok/s | Time to first token |
|---|---:|---:|
| One request, 8k in / 1k out | 122 | 775 ms |
| 8 concurrent, 2k in / 256 out | 422 | 2.7 s |
| Shared prefix, 8 concurrent | 356 | 479 ms |
| Multi-turn, 4 concurrent | 402 | 212 ms |

Accept length about 3.0. Shared-prefix cache hit 83%.

**2× joint-verify.** Output length 1,024.

| Workload | Output tok/s | Time to first token |
|---|---:|---:|
| One request, 8k in / 1k out | 126 | 2.4 s |
| 8 concurrent, 2k in / 1k out | 537 | 2.8 s |
| Shared prefix, 8 concurrent | 747 | 1.2 s |
| Multi-turn, 4 concurrent | 425 | 340 ms |

Accept length 3.7–3.9.

## Run

On a machine that already has the image and the GPUs, from this checkout:

```bash
export SGLANG_API_KEY="your-key"          # omit to leave the port open
export DATA_ROOT=/data                    # weight and log directory

# one GPU
bash serve/download_models.sh
bash serve/1x5090/dflash2_hicache.sh

# two GPUs
SERVE_PROFILE=2x5090 bash serve/download_models.sh
bash serve/2x5090/dflash2_hicache.sh
```

`download_models.sh` is idempotent. The stage script stops any previous SGLang process and starts this one. Logs go to `$DATA_ROOT/logs/sglang.log`. The OpenAI API is on port 30000 (`/v1`), model name `qwen3.8-27b`, tool parser `qwen3_coder`.

```bash
curl -s http://127.0.0.1:30000/v1/models \
  -H "Authorization: Bearer $SGLANG_API_KEY"
```

First download is about 26 GB. CUDA graph capture is quiet for a few minutes after the weights load.

## Layout

```
serve/download_models.sh          # both profiles
serve/1x5090/env.sh               # 1× pins
serve/1x5090/common.sh            # argv builder
serve/1x5090/dflash2_hicache.sh   # winning 1× stage
serve/2x5090/env.sh               # 2× pins
serve/2x5090/common.sh
serve/2x5090/dflash2_hicache.sh   # winning 2× stage
```

`MAX_TOTAL_TOKENS` on the 2× recipe is fixed at 262144. If a boot runs out of KV memory, lower concurrency, Mamba slots, or draft length. Leave the context pin where it is.
