from __future__ import annotations

import json
import shutil
from pathlib import Path

from BMM.core.paths import repo_root


ROOT = repo_root()
SOURCE_DIR = ROOT / "data" / "merge_dataset"
TARGET_DIR = ROOT / "data" / "Userdata"
TARGET_DATASET_DIR = TARGET_DIR / "datasets"

DATASET_SPECS = {
    "mit-movie_sample300": {
        "source": "mit-movie_sample300.json",
        "task": "entity_span",
        "split": "paper_eval",
    },
    "mit-restaurant-sampled300": {
        "source": "mit-restaurant-sampled300.json",
        "task": "entity_span",
        "split": "paper_eval",
    },
    "FabNER_sample300": {
        "source": "FabNER_sample300.json",
        "task": "entity_span",
        "split": "paper_eval",
    },
    "FindVehicle_sample300": {
        "source": "FindVehicle_sample300.json",
        "task": "entity_type",
        "split": "paper_eval",
    },
    "conll04_sample300": {
        "source": "conll04_sample300.json",
        "task": "relation_extraction",
        "split": "paper_eval",
    },
    "New-York-Times-RE_sample_30000_sample300": {
        "source": "New-York-Times-RE_sample_30000_sample300.json",
        "task": "relation_extraction",
        "split": "paper_eval",
    },
    "ACE05_sample300": {
        "source": "ACE05_sample300.json",
        "task": "event_extraction",
        "split": "paper_eval",
    },
    "CASIE_sample300": {
        "source": "CASIE_sample300.json",
        "task": "event_trigger",
        "split": "paper_eval",
    },
    "Ontonotes_sample300": {
        "source": "Ontonotes_sample300.json",
        "task": "entity_span",
        "split": "paper_eval",
    },
    "SciERC_sample300": {
        "source": "SciERC_sample300.json",
        "task": "relation_extraction",
        "split": "paper_eval",
    },
    "WikiNeural_sample300": {
        "source": "WikiNeural_sample300.json",
        "task": "entity_span",
        "split": "paper_eval",
    },
    "merged_dataset": {
        "source": "merged_dataset.json",
        "task": "mixed_user_batch",
        "split": "user_bundle",
    },
    "merged_dataset_25": {
        "source": "merged_dataset_25.json",
        "task": "mixed_user_batch",
        "split": "user_bundle",
    },
    "merged_dataset_50": {
        "source": "merged_dataset_50.json",
        "task": "mixed_user_batch",
        "split": "user_bundle",
    },
    "merged_dataset_75": {
        "source": "merged_dataset_75.json",
        "task": "mixed_user_batch",
        "split": "user_bundle",
    },
    "glue_merged": {
        "source": "merge_glue.jsonl",
        "task": "glue_multi_task",
        "split": "user_bundle",
        "format": "jsonl",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
}


def infer_columns(sample: dict) -> dict[str, str]:
    columns = {"response": "output"}
    if "instruction" in sample:
        columns["prompt"] = "instruction"
    if "input" in sample:
        columns["query"] = "input"
    if "history" in sample:
        columns["history"] = "history"
    return columns


def build_dataset_info() -> tuple[dict[str, dict], list[dict[str, str]]]:
    dataset_info: dict[str, dict] = {}
    manifest_items: list[dict[str, str]] = []

    for dataset_name, spec in DATASET_SPECS.items():
        source_path = SOURCE_DIR / spec["source"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source dataset: {source_path}")

        target_path = TARGET_DATASET_DIR / source_path.name
        shutil.copyfile(source_path, target_path)

        if spec.get("format") == "jsonl":
            columns = spec["columns"]
            record_count = sum(1 for _ in source_path.open("r", encoding="utf-8"))
        else:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            columns = spec.get("columns", infer_columns(payload[0] if payload else {}))
            record_count = len(payload)

        dataset_info[dataset_name] = {
            "file_name": f"datasets/{target_path.name}",
            "columns": columns,
        }
        manifest_items.append(
            {
                "dataset_name": dataset_name,
                "file_name": target_path.name,
                "task": spec["task"],
                "split": spec["split"],
                "records": str(record_count),
            }
        )

    return dataset_info, manifest_items


def write_readme(manifest_items: list[dict[str, str]]) -> None:
    paper_eval = [item for item in manifest_items if item["split"] == "paper_eval"]
    user_bundle = [item for item in manifest_items if item["split"] == "user_bundle"]

    lines = [
        "# Userdata",
        "",
        "This directory is the open-source-facing dataset bundle for the BMM framework.",
        "",
        "## Layout",
        "",
        "- `dataset_info.json`: LLaMA-Factory dataset registry for the open-source bundle.",
        "- `manifest.json`: lightweight metadata for the datasets used by the paper and demo flows.",
        "- `datasets/`: copied evaluation and user-side merge datasets.",
        "",
        "## Paper Evaluation Sets",
        "",
    ]
    for item in paper_eval:
        lines.append(f"- `{item['dataset_name']}`: `{item['file_name']}` ({item['records']} records)")

    lines.extend(
        [
            "",
            "## User Bundles",
            "",
        ]
    )
    for item in user_bundle:
        lines.append(f"- `{item['dataset_name']}`: `{item['file_name']}` ({item['records']} records)")

    lines.extend(
        [
            "",
            "## Usage",
            "",
            "Use `data/Userdata` as the `dataset_dir` when creating BMM sessions.",
        ]
    )
    (TARGET_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TARGET_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_info, manifest_items = build_dataset_info()

    with (TARGET_DIR / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2, ensure_ascii=False)

    with (TARGET_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"datasets": manifest_items}, handle, indent=2, ensure_ascii=False)

    write_readme(manifest_items)
    print(f"Wrote {TARGET_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
