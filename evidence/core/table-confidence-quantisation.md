| model | mode | dataset | distinct values | entropy (bits) | largest atom | its mass | coverage at tau0 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | 9 | 1.84 | 0.98 | 0.61 | 0.637 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | 16 | 1.72 | 0.95 | 0.73 | 0.790 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | 15 | 2.57 | 0.95 | 0.44 | 0.610 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | 19 | 3.68 | 0.80 | 0.13 | 0.577 |
| Qwen3.5-9B | non_thinking | ManyIFEval | 12 | 2.50 | 0.95 | 0.40 | 0.230 |
| Qwen3.5-9B | non_thinking | SuperGPQA | 11 | 1.84 | 1.00 | 0.63 | 0.627 |
| Qwen3.5-9B | thinking | ManyIFEval | 21 | 3.48 | 0.00 | 0.18 | 0.650 |
| Qwen3.5-9B | thinking | SuperGPQA | 17 | 2.88 | 0.95 | 0.39 | 0.693 |

Coverage is a step function of the threshold whose jumps are the confidence atoms. Where one value carries more than half the mass, a nominal 0.50 coverage target is not reachable and the achieved coverage overshoots it. That is a property of the elicited confidence channel, not a failure to optimise the threshold.
