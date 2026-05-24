# BMM

This directory is the standalone open-source layout for the BMM framework.

It packages the backend API, a lightweight frontend, paper-facing evaluation datasets, and a curated local adapter pool in one place.

## Layout

- `backend/`: backend entrypoints for the BMM API.
- `frontend/`: user-side client for creating sessions and sending control vectors.
- `api/`: FastAPI routes and request schemas.
- `core/`: merge logic, evaluation, optimization, preset loading, and path handling.
- `presets/`: open-source-ready preset definitions.
- `data/Userdata/`: copied evaluation datasets and merged user bundles.
- `adapters/pool20/`: curated local pool of 20 adapters for the open-source package.
- `configs/open_source_adapter_pool.json`: adapter manifest and preset-to-adapter subsets.
- `scripts/`: maintenance and launch scripts.

## Quick Start

### 1. Prepare open-source assets

This copies the paper-facing datasets into `data/Userdata/`, copies a curated local pool of 20 adapters into `adapters/pool20/`, and rewrites presets to use the local bundle.

```bash
cd BMM_code
python3 scripts/prepare_open_source_assets.py
```

### 2. Start the backend

```bash
cd BMM_code
python3 backend/run_server.py
```

Default backend URL:

- `http://127.0.0.1:8010`

### 3. Start the frontend

```bash
cd BMM_code/frontend
python3 -m http.server 3000
```

Open:

- `http://127.0.0.1:3000`

## BMM User Flow

1. The user picks a preset and sends `query_plus`.
2. The backend creates one session bound to one adapter subset.
3. The user sends `alpha` and `beta`.
4. The backend materializes the merged adapter and can evaluate it against `query_plus`.
5. The same session can be reused while the user swaps control vectors.

## Data

The open-source dataset root is:

- `data/Userdata`

It includes:

- paper-facing `sample300` evaluation sets
- merged user bundles such as `merged_dataset_25`, `merged_dataset_50`, `merged_dataset_75`
- `glue_merged`

## Adapters

This package intentionally ships with a curated local pool of 20 adapters so the repository stays manageable.

The full adapter collection used in the larger internal experiments will be uploaded to Hugging Face.

See:

- [adapters/README.md](adapters/README.md)
- [configs/open_source_adapter_pool.json](configs/open_source_adapter_pool.json)

## Notes

- The merge logic follows the paper's asymmetric sparsification: only `LoRA-A` is sparsified.
- Presets in this open-source package are trimmed to the curated local adapter pool.
- Model checkpoints can still be overridden when creating a session.
