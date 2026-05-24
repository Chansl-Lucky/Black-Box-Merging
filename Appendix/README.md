# Appendix Materials for Black-box Model Merging

This directory contains the online appendix material referenced from the anonymous submission. It includes implementation details, full result tables, baseline hyperparameter details, dataset/model-pool descriptions, robustness analysis, and the Evo-Merging pseudocode.

## Files

- `appendix.tex`: LaTeX appendix content with the full supplementary material.

## Paper integration

The main paper keeps compact average-result tables in the body to satisfy the page-limit constraint. Full per-task tables and diagnostic details are placed after the bibliography via:

```latex
ibliographystyle{ACM-Reference-Format}
ibliography{SIGKDD}

\clearpage
\input{appendix}
```

This ordering is important: large `table*` environments that appear before `ibliography{...}` can float into the main paper/references area and consume the 10-page body budget. Keeping the appendix input after the bibliography prevents those full tables from being counted as main-body material, subject to the conference's final page-limit rules.

## Mapping between main-text summaries and appendix tables

- Main-text Llama 3.1 average table: `tab:method_performance_merged`
  - OOD full per-task table: `tab:method_performance_ood_full`
  - ID full per-task table: `tab:method_performance_indomain_full`

- Main-text Qwen2.5-7B average table: `tab:qwen_merged_results`
  - ID full per-task table: `tab:qwen_performance_indomain`
  - OOD full per-task table: `tab:method_performance_ood_qwen`

- Main-text ablation average table: `tab:detailed_ablation_final_v2`
  - Full per-task ablation table: `tab:detailed_ablation_final_v2_full`

## Baseline pre-selection protocol

For the OOD setting, the repository contains 100+ source adapters. `Evo-Merging` and LoRAHub are evaluated on the full repository. Other white-box merging baselines use the appendix pre-selection protocol:

1. Select a smaller top-performing source-adapter subset for each target OOD task using the same validation set.
2. Run hyperparameter grid search on that curated subset.
3. Report the selected adapter sets and best hyperparameter configurations in the appendix tables.

The diagnostic oracle top-K result for `Evo-Merging` is also included to show that its full-repository behavior is not dependent on this pre-selection shortcut.

## Notes for open-sourcing

When releasing code/data, keep this appendix synchronized with any machine-readable result files. If table values are regenerated from scripts, add the scripts and raw CSV/JSON outputs here or link them from the main code repository.
