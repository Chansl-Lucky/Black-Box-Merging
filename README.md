# Black-Box-Merging

This repository contains the open-source code and supplementary materials for Black-Box Model Merging.

The main submission keeps compact average-result tables in the PDF. The larger full tables and supporting analysis are provided here so reviewers can inspect the complete evidence without searching through the repository.

## Contents

- `BMM_code/`: standalone implementation, API, frontend, presets, datasets, and curated adapter pool.
- `Appendix/post-bibliography-appendix.pdf`: the post-bibliography appendix pages with full result tables referenced by the paper.
- `Appendix/README.md`: short guide for the supplementary appendix PDF.

## Where to Find the Full Tables

All full tables referenced from the anonymous paper are collected in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf).

The PDF includes:

- baseline hyperparameter grid-search summary,
- oracle top-K source-adapter selections for OOD baselines,
- in-domain robustness results with distractor tasks,
- detailed descriptions of general model-merging baselines,
- Qwen2.5 full ID/OOD result tables,
- full per-task ablation breakdown.

## Mapping from Paper References

- Main-text `Table 4` summarizes the ablation study. The full per-task ablation table is in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf).
- Main-text `Table 5` summarizes Qwen2.5 results. The full Qwen ID and OOD tables are in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf).
- The oracle top-K pre-selection discussion points to the selected source-adapter table and grid-search table in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf).
- The robustness discussion points to the in-domain robustness table with distractor tasks in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf).
- The baseline section points to the detailed baseline descriptions in [`Appendix/post-bibliography-appendix.pdf`](Appendix/post-bibliography-appendix.pdf).

## Code

For setup and usage, see [`BMM_code/README.md`](BMM_code/README.md).
