# Black-Box-Merging

This repository hosts the anonymous code release and supplementary appendix materials for `Black-box Model Merging for Language-Model-as-a-Service with Massive Model Repositories`.

## Repository Layout

- `BMM_code/`: implementation, backend/frontend demo, presets, evaluation data, and adapter assets.
- `Appendix/post-bibliography-appendix.pdf`: supplementary appendix pages containing the full tables referenced from the paper.
- `Appendix/README.md`: appendix guide and table map.

Code setup and usage are documented in [`BMM_code/README.md`](BMM_code/README.md).

## Experimental Snapshots

### Llama 3.1 Average Results

| Method | OOD Avg Prec | OOD Avg F1 | ID Avg Prec | ID Avg F1 |
| --- | ---: | ---: | ---: | ---: |
| TIES | 24.86 | 23.81 | 33.53 | 33.51 |
| DARE | 18.26 | 17.97 | 33.75 | 32.85 |
| DELLA | 21.41 | 20.75 | 32.36 | 32.75 |
| Breadcrumbs | 36.17 | 32.25 | 35.98 | 34.34 |
| Task Arithmetic | 16.60 | 17.61 | 34.98 | 33.88 |
| EMR-Merging | 32.26 | 30.16 | 35.19 | 33.65 |
| LoRAHub | 42.72 | 40.94 | 34.70 | 34.80 |
| Evo-Merging | **53.80** | **52.13** | **39.09** | **38.03** |

### Qwen2.5-7B Average Results

| Method | OOD Avg Prec | OOD Avg F1 | ID Avg Prec | ID Avg F1 |
| --- | ---: | ---: | ---: | ---: |
| TIES | 41.22 | 32.86 | 42.86 | 41.34 |
| DARE | 37.65 | 29.65 | 43.05 | 41.57 |
| DELLA | 37.72 | 29.66 | 43.10 | 41.57 |
| Breadcrumbs | 44.13 | 36.02 | 41.71 | 40.38 |
| Task Arithmetic | 41.35 | 32.83 | 42.56 | 41.06 |
| EMR-Merging | 42.85 | 33.76 | 42.50 | 41.15 |
| LoRAHub | 57.28 | 56.28 | 42.67 | 38.42 |
| Evo-Merging | **62.45** | **61.23** | **46.93** | **43.39** |

### OOD Ablation Average Results

| Method | Avg Prec | Avg F1 |
| --- | ---: | ---: |
| Evo-Merging | **53.80** | **52.13** |
| w/o Denoising and Scaling | 23.28 | 20.75 |
| w/o Denoising | 42.72 | 40.94 |
| w/o Scaling | 26.38 | 24.27 |
| w/o Sign-flipped | 44.38 | 40.06 |

## Supplementary Materials

The full result tables referenced by the anonymous paper are collected in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf). This PDF corresponds to the post-bibliography appendix pages and includes:

- hyperparameter grid-search summaries for white-box baselines,
- oracle top-K source-adapter selections,
- robustness tables with distractor tasks,
- Qwen full-result tables,
- full ablation breakdowns.
