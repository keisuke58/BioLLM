# Seed sweep report

## Setup
- n_runs: 30 (seed_start=0)
- test_size: 0.2
- heads: gf=sgd, scgpt=sgd

## Overall (mean ± std)

| metric | Geneformer | scGPT |
|---|---:|---:|
| Accuracy | 0.694 ± 0.020 | 0.906 ± 0.012 |
| Macro F1 | 0.642 ± 0.049 | 0.886 ± 0.027 |
| Macro F1 (no Platelet) | 0.657 ± 0.021 | 0.890 ± 0.015 |

## Per-class F1 (mean ± std)

### scGPT (sorted by std desc)

| class | F1 mean | F1 std |
|---|---:|---:|
| Platelet | 0.865 | 0.121 |
| DC | 0.801 | 0.053 |
| NK | 0.861 | 0.032 |
| Mono | 0.907 | 0.026 |
| B | 0.953 | 0.015 |
| T | 0.928 | 0.012 |

### Geneformer (sorted by std desc)

| class | F1 mean | F1 std |
|---|---:|---:|
| Platelet | 0.567 | 0.271 |
| DC | 0.542 | 0.060 |
| B | 0.696 | 0.052 |
| Mono | 0.700 | 0.040 |
| NK | 0.581 | 0.036 |
| T | 0.764 | 0.018 |

## Per-class ΔF1 (Geneformer − scGPT)

### Worst classes for Geneformer (most negative ΔF1)

| rank | class | GF F1 mean | scGPT F1 mean | ΔF1 mean |
|---:|---|---:|---:|---:|
| 1 | Platelet | 0.567 | 0.865 | -0.297 |
| 2 | NK | 0.581 | 0.861 | -0.280 |
| 3 | DC | 0.542 | 0.801 | -0.259 |
| 4 | B | 0.696 | 0.953 | -0.257 |
| 5 | Mono | 0.700 | 0.907 | -0.208 |
| 6 | T | 0.764 | 0.928 | -0.164 |

### Best classes for Geneformer (most positive ΔF1)

| rank | class | GF F1 mean | scGPT F1 mean | ΔF1 mean |
|---:|---|---:|---:|---:|
| 1 | T | 0.764 | 0.928 | -0.164 |
| 2 | Mono | 0.700 | 0.907 | -0.208 |
| 3 | B | 0.696 | 0.953 | -0.257 |
| 4 | DC | 0.542 | 0.801 | -0.259 |
| 5 | NK | 0.581 | 0.861 | -0.280 |
| 6 | Platelet | 0.567 | 0.865 | -0.297 |

## Most unstable classes (scGPT std top10)

| rank | class | F1 mean | F1 std |
|---:|---|---:|---:|
| 1 | Platelet | 0.865 | 0.121 |
| 2 | DC | 0.801 | 0.053 |
| 3 | NK | 0.861 | 0.032 |
| 4 | Mono | 0.907 | 0.026 |
| 5 | B | 0.953 | 0.015 |
| 6 | T | 0.928 | 0.012 |

## Notes
- `Platelet` は support が非常に小さく、seed/splitで F1 が大きくブレやすいです。
