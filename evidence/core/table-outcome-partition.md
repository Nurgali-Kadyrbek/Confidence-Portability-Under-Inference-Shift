| model | mode | dataset | shift | dS | dE | dq | da_C | da_tau | dK |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | -0.067 | -0.255 | +0.000 | +0.000 | +0.322 | -0.322 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | +0.025 | -0.008 | +0.000 | +0.000 | -0.017 | +0.017 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | -0.120 | -0.237 | +0.000 | +0.000 | +0.357 | -0.357 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | -0.002 | +0.005 | +0.000 | +0.000 | -0.003 | +0.003 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | -0.018 | -0.053 | +0.000 | -0.007 | +0.078 | -0.072 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | -0.005 | +0.058 | +0.000 | +0.022 | -0.075 | +0.053 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | -0.017 | +0.053 | -0.017 | -0.037 | +0.017 | +0.037 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | -0.087 | -0.055 | +0.153 | +0.085 | -0.097 | -0.142 |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | +0.007 | +0.027 | +0.000 | +0.000 | -0.033 | +0.033 |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | +0.003 | -0.010 | +0.000 | +0.000 | +0.007 | -0.007 |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | -0.047 | -0.018 | +0.000 | +0.000 | +0.065 | -0.065 |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | +0.017 | +0.012 | +0.000 | +0.000 | -0.028 | +0.028 |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | -0.003 | +0.090 | +0.000 | -0.005 | -0.082 | +0.087 |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | -0.003 | -0.110 | +0.000 | +0.205 | -0.092 | -0.113 |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | -0.033 | -0.023 | +0.002 | -0.002 | +0.057 | -0.057 |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | +0.053 | +0.030 | +0.000 | +0.008 | -0.092 | +0.083 |

S accepted-correct, E accepted-wrong, q answer failure, a_C confidence failure, a_tau low-confidence abstention. S + E + q + a_C + a_tau = 1 and K = S + E.
