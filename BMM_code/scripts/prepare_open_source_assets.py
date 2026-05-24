from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from BMM.core.paths import repo_root
from BMM.scripts.organize_userdata import main as organize_userdata


ROOT = repo_root()
LEGACY_ROOT = ROOT.parent
TARGET_ADAPTER_ROOT = ROOT / "adapters" / "pool20"
CONFIG_DIR = ROOT / "configs"
PRESET_DIR = ROOT / "presets"

CURATED_ADAPTERS = {
    "NER_mit-movie_sft": LEGACY_ROOT / "adapters" / "halflora" / "NER_mit-movie_sft",
    "ET_mit-movie_sft": LEGACY_ROOT / "adapters" / "halflora" / "ET_mit-movie_sft",
    "NER_FabNER_sft": LEGACY_ROOT / "adapters" / "halflora" / "NER_FabNER_sft",
    "ET_FabNER_sft": LEGACY_ROOT / "adapters" / "halflora" / "ET_FabNER_sft",
    "NER_FindVehicle_sft": LEGACY_ROOT / "adapters" / "halflora" / "NER_FindVehicle_sft",
    "ET_FindVehicle_sft": LEGACY_ROOT / "adapters" / "halflora" / "ET_FindVehicle_sft",
    "NER_mit-restaurant_sft": LEGACY_ROOT / "adapters" / "halflora" / "NER_mit-restaurant_sft",
    "ET_mit-restaurant_sft": LEGACY_ROOT / "adapters" / "halflora" / "ET_mit-restaurant_sft",
    "RE_conll04": LEGACY_ROOT / "adapters" / "halflora" / "RE_conll04",
    "EPR_conll04_sample_5000_sft": LEGACY_ROOT / "adapters" / "halflora" / "EPR_conll04_sample_5000_sft",
    "EP_conll04_sft": LEGACY_ROOT / "adapters" / "halflora" / "EP_conll04_sft",
    "RE_New-York-Times-RE_sample_30000_sft": LEGACY_ROOT / "adapters" / "halflora" / "RE_New-York-Times-RE_sample_30000_sft",
    "EPR_New-York-Times-RE_sample_30000_sft": LEGACY_ROOT / "adapters" / "halflora" / "EPR_New-York-Times-RE_sample_30000_sft",
    "cola_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "cola_qa_train",
    "mnli_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "mnli_qa_train",
    "mrpc_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "mrpc_qa_train",
    "qnli_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "qnli_qa_train",
    "qqp_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "qqp_qa_train",
    "rte_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "rte_qa_train",
    "sst2_qa_train": LEGACY_ROOT / "adapters" / "halflora" / "sst2_qa_train",
}

PRESET_SUBSETS = {
    "mit_movie": [
        "NER_mit-movie_sft",
        "ET_mit-movie_sft",
        "NER_mit-restaurant_sft",
        "ET_mit-restaurant_sft",
        "NER_FabNER_sft",
        "ET_FabNER_sft",
    ],
    "fabner": [
        "NER_FabNER_sft",
        "ET_FabNER_sft",
        "NER_FindVehicle_sft",
        "ET_FindVehicle_sft",
        "NER_mit-movie_sft",
        "ET_mit-movie_sft",
    ],
    "findvehicle": [
        "NER_FindVehicle_sft",
        "ET_FindVehicle_sft",
        "NER_FabNER_sft",
        "ET_FabNER_sft",
        "NER_mit-movie_sft",
        "ET_mit-movie_sft",
    ],
    "mit_restaurant": [
        "NER_mit-restaurant_sft",
        "ET_mit-restaurant_sft",
        "NER_mit-movie_sft",
        "ET_mit-movie_sft",
        "NER_FabNER_sft",
        "ET_FabNER_sft",
    ],
    "conll04": [
        "RE_conll04",
        "EPR_conll04_sample_5000_sft",
        "EP_conll04_sft",
        "RE_New-York-Times-RE_sample_30000_sft",
        "EPR_New-York-Times-RE_sample_30000_sft",
    ],
    "new_york_times": [
        "RE_New-York-Times-RE_sample_30000_sft",
        "EPR_New-York-Times-RE_sample_30000_sft",
        "RE_conll04",
        "EPR_conll04_sample_5000_sft",
        "EP_conll04_sft",
    ],
    "glue_fhx": [
        "cola_qa_train",
        "mnli_qa_train",
        "mrpc_qa_train",
        "qnli_qa_train",
        "qqp_qa_train",
        "rte_qa_train",
        "sst2_qa_train",
    ],
}


def copy_curated_adapters() -> list[dict[str, str]]:
    TARGET_ADAPTER_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for adapter_name, source_dir in CURATED_ADAPTERS.items():
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing adapter source: {source_dir}")
        target_dir = TARGET_ADAPTER_ROOT / adapter_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        manifest.append(
            {
                "adapter_name": adapter_name,
                "path": f"adapters/pool20/{adapter_name}",
            }
        )
    return manifest


def rewrite_presets() -> None:
    for preset_id, adapter_names in PRESET_SUBSETS.items():
        preset_path = PRESET_DIR / f"{preset_id}.json"
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
        payload["adapter_dict"] = {
            adapter_name: f"adapters/pool20/{adapter_name}/adapter_model.safetensors"
            for adapter_name in adapter_names
        }
        payload.setdefault("metadata", {})
        payload["metadata"]["open_source_adapter_pool"] = "pool20"
        payload["metadata"]["open_source_note"] = (
            "This preset ships with a curated local pool of 20 adapters. "
            "The full adapter collection will be released on Hugging Face."
        )
        preset_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_adapter_docs(adapter_manifest: list[dict[str, str]]) -> None:
    adapter_dir = ROOT / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    readme_lines = [
        "# Adapter Bundle",
        "",
        "This open-source package includes a curated local pool of 20 adapters under `adapters/pool20/`.",
        "",
        "The complete adapter pool used during the full project will be uploaded to Hugging Face.",
        "",
        "## Included Local Adapters",
        "",
    ]
    for item in adapter_manifest:
        readme_lines.append(f"- `{item['adapter_name']}`: `{item['path']}`")
    (adapter_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with (CONFIG_DIR / "open_source_adapter_pool.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pool_name": "pool20",
                "adapter_count": len(adapter_manifest),
                "adapters": adapter_manifest,
                "presets": PRESET_SUBSETS,
                "huggingface_note": "The full adapter collection will be uploaded to Hugging Face.",
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    organize_userdata()
    adapter_manifest = copy_curated_adapters()
    rewrite_presets()
    write_adapter_docs(adapter_manifest)
    print(f"Prepared open-source assets under {ROOT}")


if __name__ == "__main__":
    main()
