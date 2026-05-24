# Userdata

This directory is the open-source-facing dataset bundle for the BMM framework.

## Layout

- `dataset_info.json`: LLaMA-Factory dataset registry for the open-source bundle.
- `manifest.json`: lightweight metadata for the datasets used by the paper and demo flows.
- `datasets/`: copied evaluation and user-side merge datasets.

## Paper Evaluation Sets

- `mit-movie_sample300`: `mit-movie_sample300.json` (300 records)
- `mit-restaurant-sampled300`: `mit-restaurant-sampled300.json` (300 records)
- `FabNER_sample300`: `FabNER_sample300.json` (300 records)
- `FindVehicle_sample300`: `FindVehicle_sample300.json` (300 records)
- `conll04_sample300`: `conll04_sample300.json` (300 records)
- `New-York-Times-RE_sample_30000_sample300`: `New-York-Times-RE_sample_30000_sample300.json` (300 records)
- `ACE05_sample300`: `ACE05_sample300.json` (300 records)
- `CASIE_sample300`: `CASIE_sample300.json` (300 records)
- `Ontonotes_sample300`: `Ontonotes_sample300.json` (300 records)
- `SciERC_sample300`: `SciERC_sample300.json` (300 records)
- `WikiNeural_sample300`: `WikiNeural_sample300.json` (300 records)

## User Bundles

- `merged_dataset`: `merged_dataset.json` (800 records)
- `merged_dataset_25`: `merged_dataset_25.json` (200 records)
- `merged_dataset_50`: `merged_dataset_50.json` (350 records)
- `merged_dataset_75`: `merged_dataset_75.json` (525 records)
- `glue_merged`: `merge_glue.jsonl` (500 records)

## Usage

Use `data/Userdata` as the `dataset_dir` when creating BMM sessions.
