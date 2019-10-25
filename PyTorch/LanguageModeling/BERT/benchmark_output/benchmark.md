## training setting: fp16 + tensorcore (t4) + fused kernel

| card | token throughput per sec | per token time in ms | number of tokens per epoch | total epochs | estimated time per epoch (1M1G)  |
|---|---|---|---|---|---|
| p100 | 3228.8 | 0.3097 ms | 16,752,705,536 | 40 | 1441.6 hours (~60 days) |
| t4 | 5490.6 | 0.18213 ms | 16,752,705,536 | 40 | 847.6 hours (~35 days) |