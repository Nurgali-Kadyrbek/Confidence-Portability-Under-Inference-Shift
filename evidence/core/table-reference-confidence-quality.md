| model | mode | dataset | accuracy | mean C | gap | AUROC (95% CI) | excludes 0.5 | BSS (95% CI) | MCB/UNC | DSC/UNC |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | ManyIFEval | 0.302 | 0.930 | +0.628 | 0.447 [0.404, 0.489] | yes | -1.952 [-2.362, -1.628] | 1.967 | 0.016 |
| Ministral-3-8B-Instruct-2512-BF16 | instruct | SuperGPQA | 0.302 | 0.894 | +0.592 | 0.539 [0.500, 0.577] | no | -1.795 [-2.175, -1.482] | 1.803 | 0.008 |
| Ministral-3-8B-Reasoning-2512 | reasoning | ManyIFEval | 0.125 | 0.873 | +0.748 | 0.527 [0.467, 0.587] | no | -5.447 [-7.145, -4.302] | 5.456 | 0.009 |
| Ministral-3-8B-Reasoning-2512 | reasoning | SuperGPQA | 0.373 | 0.614 | +0.240 | 0.589 [0.542, 0.637] | yes | -0.524 [-0.694, -0.376] | 0.566 | 0.042 |
| Qwen3.5-9B | non_thinking | ManyIFEval | 0.543 | 0.888 | +0.344 | 0.671 [0.630, 0.711] | yes | -0.463 [-0.591, -0.351] | 0.564 | 0.101 |
| Qwen3.5-9B | non_thinking | SuperGPQA | 0.375 | 0.934 | +0.559 | 0.550 [0.510, 0.591] | yes | -1.450 [-1.726, -1.218] | 1.463 | 0.013 |
| Qwen3.5-9B | thinking | ManyIFEval | 0.053 | 0.361 | +0.308 | 0.873 [0.835, 0.905] | yes | -3.894 [-6.294, -2.537] | 4.022 | 0.128 |
| Qwen3.5-9B | thinking | SuperGPQA | 0.565 | 0.879 | +0.314 | 0.645 [0.602, 0.689] | yes | -0.411 [-0.528, -0.299] | 0.488 | 0.078 |

Reference coordinate, held-out test partition. Secondary and descriptive: these eight comparisons are not in the confirmatory family and carry no multiplicity correction, so an interval that excludes 0.5 by a narrow margin is weak evidence of reversed ranking on its own.

BSS = (DSC - MCB)/UNC against the base-rate forecast, so a negative skill score says miscalibration outweighs whatever discrimination the forecast carries. Both ratios are shown because only their difference appears in the skill score and they are separate failures: a forecast that ranks well can still be worthless as a probability.
