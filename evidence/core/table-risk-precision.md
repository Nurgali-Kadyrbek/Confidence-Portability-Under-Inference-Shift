| model | mode | dataset | shift | accepted n | CI width | band (+-0.025) | could fit |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | 382 | 0.128 | 2.6x | **no** |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | 382 | 0.050 | 1.0x | **no** |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | 474 | 0.097 | 1.9x | **no** |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | 474 | 0.015 | 0.3x | yes |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | 366 | 0.069 | 1.4x | **no** |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | 366 | 0.063 | 1.3x | **no** |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | 346 | 0.083 | 1.7x | **no** |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | 346 | 0.110 | 2.2x | **no** |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | 138 | 0.180 | 3.6x | **no** |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | 138 | 0.156 | 3.1x | **no** |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | 376 | 0.063 | 1.3x | **no** |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | 376 | 0.045 | 0.9x | yes |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | 390 | 0.020 | 0.4x | yes |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | 390 | 0.026 | 0.5x | yes |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | 416 | 0.073 | 1.5x | **no** |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | 416 | 0.072 | 1.4x | **no** |

For 12 of 16 contrasts the realised 95% interval was wider than the entire prespecified equivalence region, so it could not have fallen inside that region wherever it was centred. This describes the precision achieved here, not a bound on what a sample of this size could achieve: a different true risk, with the same N, would give a different width. The reason the achieved precision is what it is: selective risk is estimated on accepted items, so its effective sample size is NK rather than N.
