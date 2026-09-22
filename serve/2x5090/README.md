# 2× RTX 5090

Winning stage: `dflash2_hicache`. Tensor parallel 2.

| File | Role |
|---|---|
| `env.sh` | Pins. Concurrency 8, Mamba 96, draft 6, graphs 16, HiCache 1, mem-frac 0.86, token pin 262144 |
| `common.sh` | Builds the SGLang argv and starts the process |
| `dflash2_hicache.sh` | NVFP4 + DFlash2 + HiCache |

Measured context: **262,144** tokens. `MAX_TOTAL_TOKENS` is hardcoded to that window.

```bash
SERVE_PROFILE=2x5090 bash serve/download_models.sh
bash serve/2x5090/dflash2_hicache.sh
```

Look for a host with at least 128 GB of RAM so the HiCache pool fits.
