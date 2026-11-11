# Ours (DeepSeek) vs Paper (GPT-4o-mini), LoCoMo first 2 conversations

Paper numbers: Table 1 (A-Mem row) and Table 3 (w/o ME row). NOTE: different backbone LLM — not directly comparable.

| Category | Paper full F1 | Ours full F1 | Paper w/o ME F1 | Ours w/o ME F1 | Paper full BLEU | Ours full BLEU |
|---|---|---|---|---|---|---|
| Multi-Hop | 27.02 | 33.73 | 21.35 | 29.07 | 20.09 | 21.84 |
| Temporal | 45.85 | 28.90 | 31.24 | 31.23 | 36.67 | 24.03 |
| Open-Domain | 12.14 | 16.77 | 10.13 | 15.99 | 12.00 | 10.79 |
| Single-Hop | 44.65 | 37.54 | 39.17 | 39.24 | 37.06 | 32.13 |
| Adversarial | 50.03 | 31.38 | 44.16 | 27.08 | 49.47 | 30.58 |