| model | mode | dataset | shift | I_ans | I_conf | dq | dK | dR (95% CI) | dS | Holm p | class |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | 1.00 | 0.82 | +0.000 | -0.322 | -0.052 [-0.118, +0.010] | -0.067 | 0.876 | inconclusive |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | 0.71 | 0.21 | +0.000 | +0.017 | -0.030 [-0.056, -0.005] | +0.025 | 0.234 | inconclusive |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | 0.36 | 0.74 | +0.000 | -0.357 | +0.011 [-0.037, +0.059] | -0.120 | 1.000 | inconclusive |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | 0.01 | 0.01 | +0.000 | +0.003 | +0.003 [-0.004, +0.011] | -0.002 | 1.000 | equivalent |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | 1.00 | 0.86 | +0.000 | -0.072 | +0.017 [-0.018, +0.051] | -0.018 | 1.000 | inconclusive |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | 1.00 | 0.74 | +0.000 | +0.053 | +0.018 [-0.013, +0.050] | -0.005 | 1.000 | inconclusive |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | 0.83 | 0.56 | -0.017 | +0.037 | +0.053 [+0.012, +0.095] | -0.017 | 0.171 | inconclusive |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | 0.86 | 0.72 | +0.153 | -0.142 | +0.045 [-0.009, +0.102] | -0.087 | 1.000 | inconclusive |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | 1.00 | 0.79 | +0.000 | +0.033 | +0.067 [-0.023, +0.156] | +0.007 | 1.000 | inconclusive |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | 0.93 | 0.58 | +0.000 | -0.007 | -0.036 [-0.112, +0.044] | +0.003 | 1.000 | inconclusive |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | 0.22 | 0.29 | +0.000 | -0.065 | +0.036 [+0.005, +0.067] | -0.047 | 0.302 | inconclusive |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | 0.06 | 0.12 | +0.000 | +0.028 | -0.008 [-0.030, +0.015] | +0.017 | 1.000 | inconclusive |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | 1.00 | 1.00 | +0.000 | +0.087 | +0.016 [+0.007, +0.026] | -0.003 | 0.003 | change |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | 1.00 | 1.00 | +0.000 | -0.113 | -0.013 [-0.027, -0.001] | -0.003 | 0.306 | inconclusive |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | 1.00 | 1.00 | +0.002 | -0.057 | -0.004 [-0.040, +0.033] | -0.033 | 1.000 | inconclusive |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | 1.00 | 1.00 | +0.000 | +0.083 | -0.001 [-0.037, +0.034] | +0.053 | 1.000 | inconclusive |
