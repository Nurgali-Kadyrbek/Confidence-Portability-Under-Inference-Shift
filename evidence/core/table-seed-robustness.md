| model | mode | dataset | shift | endpoint | per-seed | sd | range | sign consistent | seed share of variance |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | dR | +0.092/-0.107/-0.008 | 0.081 | 0.199 | **no** | - |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | dK | -0.383/-0.156/-0.297 | 0.093 | 0.227 | yes | 0.56 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | high-temperature | dS | -0.148/+0.000/-0.109 | 0.063 | 0.148 | yes | 0.39 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | dR | +0.009/-0.031/-0.003 | 0.017 | 0.040 | **no** | - |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | dK | -0.008/-0.008/+0.016 | 0.011 | 0.023 | **no** | 0.62 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | low-top-p | dS | -0.008/+0.016/+0.008 | 0.010 | 0.023 | **no** | 0.66 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | dR | +0.039/+0.037/+0.046 | 0.004 | 0.009 | yes | - |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | dK | -0.383/-0.375/-0.367 | 0.006 | 0.016 | yes | 0.46 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | high-temperature | dS | -0.148/-0.141/-0.141 | 0.004 | 0.008 | yes | 0.54 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | dR | +0.000/+0.000/-0.010 | 0.005 | 0.010 | yes | - |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | dK | +0.000/+0.000/+0.000 | 0.000 | 0.000 | yes | - |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | low-top-p | dS | +0.000/+0.000/+0.008 | 0.004 | 0.008 | yes | 0.67 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | dR | -0.050/-0.039/-0.068 | 0.012 | 0.029 | yes | - |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | dK | -0.078/-0.055/+0.047 | 0.054 | 0.125 | **no** | 0.71 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | high-temperature | dS | +0.016/+0.016/+0.047 | 0.015 | 0.031 | yes | 0.53 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | dR | -0.029/-0.045/-0.020 | 0.010 | 0.025 | yes | - |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | dK | +0.039/+0.094/+0.094 | 0.026 | 0.055 | yes | 0.64 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | low-top-p | dS | +0.023/+0.039/+0.023 | 0.007 | 0.016 | yes | 0.17 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | dR | +0.056/+0.037/+0.084 | 0.020 | 0.047 | yes | - |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | dK | +0.070/+0.000/-0.008 | 0.035 | 0.078 | **no** | 0.67 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | high-temperature | dS | +0.000/-0.023/-0.055 | 0.022 | 0.055 | yes | 0.50 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | dR | +0.020/-0.004/+0.026 | 0.013 | 0.030 | **no** | - |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | dK | -0.086/-0.055/-0.094 | 0.017 | 0.039 | yes | 0.59 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | low-top-p | dS | -0.055/-0.023/-0.062 | 0.017 | 0.039 | yes | 0.68 |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | dR | +0.081/-0.098/+0.182 | 0.116 | 0.280 | **no** | - |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | dK | -0.016/+0.031/+0.086 | 0.042 | 0.102 | **no** | 0.67 |
| Qwen3.5-9B | non_thinking | ManyIFEval | high-temperature | dS | -0.031/+0.047/+0.023 | 0.033 | 0.078 | **no** | 0.59 |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | dR | -0.040/-0.144/+0.079 | 0.091 | 0.223 | **no** | - |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | dK | -0.031/+0.008/+0.078 | 0.045 | 0.109 | **no** | 0.69 |
| Qwen3.5-9B | non_thinking | ManyIFEval | low-top-p | dS | -0.016/+0.039/+0.047 | 0.028 | 0.062 | **no** | - |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | dR | +0.062/+0.068/+0.023 | 0.020 | 0.045 | yes | - |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | dK | -0.086/-0.070/-0.023 | 0.027 | 0.062 | yes | 0.58 |
| Qwen3.5-9B | non_thinking | SuperGPQA | high-temperature | dS | -0.062/-0.062/-0.023 | 0.018 | 0.039 | yes | 0.58 |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | dR | -0.008/-0.014/+0.006 | 0.008 | 0.019 | **no** | - |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | dK | +0.055/+0.062/+0.047 | 0.006 | 0.016 | yes | 0.55 |
| Qwen3.5-9B | non_thinking | SuperGPQA | low-top-p | dS | +0.023/+0.031/+0.016 | 0.006 | 0.016 | yes | 0.53 |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | dR | +0.005/-0.021/+0.002 | 0.011 | 0.026 | **no** | - |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | dK | +0.039/-0.141/+0.016 | 0.080 | 0.180 | **no** | 0.62 |
| Qwen3.5-9B | thinking | ManyIFEval | high-temperature | dS | +0.000/-0.000/+0.000 | 0.000 | 0.000 | yes | - |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | dR | -0.025/-0.006/-0.024 | 0.008 | 0.018 | yes | - |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | dK | -0.203/-0.219/-0.148 | 0.030 | 0.070 | yes | 0.66 |
| Qwen3.5-9B | thinking | ManyIFEval | low-top-p | dS | -0.008/-0.016/-0.000 | 0.006 | 0.016 | yes | - |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | dR | -0.001/-0.068/-0.067 | 0.032 | 0.067 | yes | - |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | dK | -0.062/-0.055/-0.055 | 0.004 | 0.008 | yes | 0.62 |
| Qwen3.5-9B | thinking | SuperGPQA | high-temperature | dS | -0.039/+0.016/+0.016 | 0.026 | 0.055 | **no** | 0.78 |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | dR | +0.007/-0.082/+0.008 | 0.042 | 0.090 | **no** | - |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | dK | +0.008/+0.062/+0.023 | 0.023 | 0.055 | yes | 0.68 |
| Qwen3.5-9B | thinking | SuperGPQA | low-top-p | dS | +0.000/+0.102/+0.008 | 0.046 | 0.102 | yes | 0.50 |

18 of 48 reported contrast-endpoint combinations changed sign across seeds. Selective risk carries no variance split: it is a ratio with no per-item value, so decomposing it would restrict to items accepted at two or more seeds.

Across the 68 combinations that admit a variance split, the seed share exceeded one half in 62 of them (median 0.642, range 0.171 to 0.783). This is a panel-level description, not a property of every contrast: 6 combinations are dominated by item heterogeneity instead. Three seeds on 128 items cannot support a stronger claim than that within-item variation across decoder realisations was frequently the larger of the two components.

This is a 128-item subset, so its per-seed point estimates are noisier than the 600-item held-out estimates and do not replace them. All three seeds were generated in one run because generation is not invariant to batch composition, so what is measured is the variability of a rerun under one batching regime.
