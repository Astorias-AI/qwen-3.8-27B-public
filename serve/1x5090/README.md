# 1× RTX 5090

Winning stage: `dflash2_hicache`.

| File | Role |
|---|---|
| `env.sh` | Pins. Concurrency 4, Mamba 12, draft 4, graphs 4, mem-frac 0.90, token pin 65536 |
| `common.sh` | Builds the SGLang argv and starts the process |
| `dflash2_hicache.sh` | NVFP4 + DFlash2 + HiCache |

Effective KV pool on this card: **28,192** tokens. HiCache ratio 2 spills prefix KV to host RAM. GDN state stays on the GPU.

```bash
bash serve/download_models.sh
bash serve/1x5090/dflash2_hicache.sh
```
