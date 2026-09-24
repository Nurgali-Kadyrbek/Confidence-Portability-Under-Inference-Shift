# CPIS — Confidence Portability Under Inference Shift

Code, configurations and frozen result artifacts for the paper

> **Portability of Confidence-Threshold Policies Under
> Inference-Configuration Shift in Open-Weight Language Models**

The study asks whether a confidence-threshold decision policy selected under
one inference configuration stays statistically reliable when decoding settings
change. It evaluates four model conditions across two open-weight families on
two benchmarks under three inference configurations, with every item evaluated
under every configuration.

There are two ways to use this repository, and they cost very different amounts.

| | what it needs | roughly |
| --- | --- | --- |
| **A. Reproduce the analysis** from the released artifacts | CPU, a Python environment | minutes |
| **B. Reproduce the generations** from the models | 8×48 GB GPUs, the benchmark corpora, the model weights | about a day |

Path A regenerates every number, table and figure in the paper. Start there.

---

## A. Reproduce the analysis from released artifacts

### 1. System requirements

* Linux, Python 3.11 or newer
* No GPU
* About 200 MB of disk

### 2. Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

`pyproject.toml` declares the dependencies; `uv.lock` pins the exact versions
used for the published results if you prefer `uv sync`.

### 3. Check the artifacts are the ones behind the paper

```bash
sha256sum -c MANIFEST.sha256
```

### 4. Regenerate the tables and figures

The published result artifacts live in `evidence/`. Nothing below invokes a
model; the analysis code cannot, by design.

```bash
# every LaTeX macro, table and figure used by the manuscript
PYTHONPATH=src python scripts/make_manuscript_inputs.py
PYTHONPATH=src python scripts/make_manuscript_figures.py

# the standalone Markdown report tables and SVG figures
PYTHONPATH=src python scripts/report_core_study.py \
  --thresholds evidence/protocol/calibration_thresholds_v1.json \
  --analysis   evidence/core/held-out-analysis.json
```

Outputs land in `paper/generated/` and `evidence/core/`. They should match the
files already present; `git diff` after running is the check.

### 5. Recompute the analysis from the parsed records

This step needs the record store (see Path B) because it reads the per-item
generations. With `CPIS_STORAGE_ROOT` pointing at that store:

```bash
export CPIS_STORAGE_ROOT=/path/to/record/store

PYTHONPATH=src python scripts/freeze_calibration_thresholds.py \
  --destination evidence/protocol/calibration_thresholds_v1.json

PYTHONPATH=src python scripts/analyze_core_test.py \
  --thresholds evidence/protocol/calibration_thresholds_v1.json \
  --destination evidence/core/held-out-analysis.json

PYTHONPATH=src python scripts/reference_confidence_quality.py \
  --configs configs/core/test --condition reference \
  --partition-label "held-out test" \
  --destination evidence/core/reference-confidence-quality-heldout.json

PYTHONPATH=src python scripts/measurement_properties.py \
  --thresholds evidence/protocol/calibration_thresholds_v1.json \
  --analysis evidence/core/held-out-analysis.json \
  --destination evidence/core/measurement-properties.json

PYTHONPATH=src python scripts/analyze_seed_robustness.py \
  --thresholds evidence/protocol/calibration_thresholds_v1.json \
  --destination evidence/core/seed-robustness.json
```

All of these are deterministic: the bootstrap and randomisation seeds are fixed
in `configs/analysis/core-v5.yaml`, so reruns reproduce the published values
exactly.

### 6. Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

The suite covers the scoring and parsing contracts, the outcome partition, the
threshold rule, the bootstrap's pairing, the classification precedence and the
CORP decomposition.

### 7. Build the manuscript

```bash
cd paper && ./build.sh cpis yes
```

This needs a TeX distribution and the official MDPI class files, which are not
redistributed here; see `paper/TEMPLATE_PROVENANCE.md` for exactly which
template version was used and where to obtain it.

---

## B. Reproduce the generations from scratch

### 1. Requirements

* 8× NVIDIA L40 48 GB (or equivalent); the study is BF16, one resident model
  replica per device, no offload
* The benchmark corpora and the model weights, obtained from their original
  providers

### 2. Obtain the data and models

Neither the benchmark corpora nor the model weights are redistributed here.
Both are pinned by revision so the exact inputs can be reconstructed:

| dataset | upstream revision | partition | prepared file | SHA-256 |
| --- | --- | --- | --- | --- |
| `m-a-p/SuperGPQA` | `4430d4458112c7d4497fdcf94d7cc223313d6acf` | development | `prepared/supergpqa-v2-sealed/items-development.jsonl` | `9938ae277c8444ec6c548b5b72d7cda0d4b735fff85487d4fc6fcecebccfe23e` |
| `m-a-p/SuperGPQA` | `4430d4458112c7d4497fdcf94d7cc223313d6acf` | test | `prepared/supergpqa-v2-sealed/items-test.jsonl` | `2a438eb6e085b4ae01da0bb36542d7ecdc487fe7e5fd1f5893ab1c247775783c` |
| `kenoharada/Multiple-Instructions-Following:ManyIFEval` | `5c1ba29e631e2bbee981ebd620898c2750c5ddbf` | development | `prepared/manyifeval-v1-sealed/items-development.jsonl` | `3149a4b4f8afa08a6eec3ea6c3586300ca82fea00b7bb03c2568d23d74467871` |
| `kenoharada/Multiple-Instructions-Following:ManyIFEval` | `5c1ba29e631e2bbee981ebd620898c2750c5ddbf` | test | `prepared/manyifeval-v1-sealed/items-test.jsonl` | `35dc185205c53a2e96c4945de7d2fd8d0692d55dbd431ce528b6b570ae4da949` |

Model weights, tokenizers and chat templates:

| model | revision | tokenizer revision |
| --- | --- | --- |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | same |
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f6...` | same |
| `mistralai/Ministral-3-8B-Reasoning-2512` | `81eaece1948f38...` | same |

The full records, including chat-template identifiers and their checksums, are
in `integrations/`. The prepared partition files are produced deterministically
from the upstream corpora; after preparation their checksums must match the
table above, and the pipeline refuses to run if they do not.

### 3. Point the code at a record store

```bash
export CPIS_STORAGE_ROOT=/path/with/space/for/records
```

Raw generations are written there once, as read-only files, before any parsing.

### 4. Run a phase

Each experiment is fully described by its configuration file; there are no
command-line knobs that change the science.

```bash
PYTHONPATH=src python -m cpis.cli run-matrix \
  --config configs/core/calibration/qwen35-9b-supergpqa-calibration-v1.yaml \
  --repository-root . --replica-index 0
```

Held-out and seed-robustness configurations read the sealed test partition,
which is guarded by a phase gate: the run refuses to start unless the protocol
files and configurations named in `gates/test-open.yaml` are committed and
byte-identical to their recorded checksums. Rebuild the gate for your own
checkout with `scripts/build_test_gate.py`.

ManyIFEval is scored by its official evaluator in a separate step:

```bash
PYTHONPATH=src python -m cpis.cli score-matrix \
  --config configs/core/test/qwen35-9b-manyifeval-test-v1.yaml \
  --repository-root .
```

Generation is **not** bit-reproducible across different batch compositions on
this engine; `evidence/environment/batch_size_generation_invariance_v1.json`
records the measurement. Reproducing the published records byte for byte
therefore requires the same batch size and item ordering, which the
configurations fix.

---

## Layout

```
configs/      experiment and analysis definitions; one file per reported run
integrations/ pinned dataset and model records, with partition checksums
evidence/     frozen analysis artifacts (the inputs to every table and figure)
  core/         held-out analysis, reference confidence quality, measurement
                properties, seed robustness, and the rendered report tables
  protocol/     calibration thresholds and the study freeze record
  environment/  batch-invariance and confidence-budget evidence
scripts/      preparation, analysis and reporting entry points
src/cpis/     the library
paper/        manuscript source and the generated tables, figures and macros
tests/        the test suite
```

## Notes on what is and is not here

The library ships whole, including modules for a separate agent-based
investigation that this paper does not use; removing them would break the
command line interface that generation and official scoring depend on. The
configurations and results for that investigation, and for superseded
development campaigns, are not included.

## Licence

Code is released under the MIT licence (`LICENSE`). The benchmark corpora and
the model weights are covered by their own licences from their original
providers and are not redistributed here.

## Citation

See `CITATION.cff`.
