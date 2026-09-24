| model | mode | dataset | shift | I_ans | I_conf | I_conf given answer identical | n |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | 1.00 | 0.82 | undefined | 0 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | 0.71 | 0.21 | 0.006 | 176 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | 0.36 | 0.74 | 0.723 | 382 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | 0.01 | 0.01 | 0.013 | 594 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | 1.00 | 0.86 | undefined | 0 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | 1.00 | 0.74 | undefined | 0 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | 0.83 | 0.56 | 0.442 | 104 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | 0.86 | 0.72 | 0.481 | 81 |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | 1.00 | 0.79 | 0.500 | 2 |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | 0.93 | 0.58 | 0.119 | 42 |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | 0.22 | 0.29 | 0.242 | 467 |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | 0.06 | 0.12 | 0.105 | 563 |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | 1.00 | 1.00 | undefined | 0 |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | 1.00 | 1.00 | undefined | 0 |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | 1.00 | 1.00 | undefined | 0 |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | 1.00 | 1.00 | undefined | 0 |

A manipulation diagnostic, not a causal decomposition. The conditional column is identified only on items the shift left alone, and that subset is enriched for items the model answers stably rather than being a random sample.

7 of these contrasts changed every compared answer, leaving no items on which to condition; the question cannot be asked of them in this design.
