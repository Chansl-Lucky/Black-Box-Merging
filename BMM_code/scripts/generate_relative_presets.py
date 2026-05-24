from __future__ import annotations

import json
from pathlib import Path

from BMM.core.paths import repo_root, sanitize_path


ROOT = repo_root()
PRESET_DIR = ROOT / "bmm_service" / "presets"

SOURCES = {
    "mit_movie": ROOT / "adapters" / "bllama_halflora" / "mit_movie.json",
    "fabner": ROOT / "adapters" / "bllama_halflora" / "fabner.json",
    "findvehicle": ROOT / "adapters" / "bllama_halflora" / "findvehicle.json",
    "conll04": ROOT / "adapters" / "bllama_halflora" / "conll04.json",
    "new_york_times": ROOT / "adapters" / "bllama_halflora" / "new_york_times.json",
    "mit_restaurant": ROOT / "adapters" / "bllama_halflora" / "mit_restaurant.json",
    "glue_fhx": ROOT / "adapters" / "FHX.json",
}


def infer_query_plus(preset_id: str, payload: dict) -> dict:
    dataset_name = payload.get("dataset_name")
    if preset_id == "glue_fhx":
        dataset_name = "glue_merged"
    return {
        "dataset": dataset_name or preset_id,
        "dataset_dir": "data",
        "template": "llama3",
        "cutoff_len": 2048,
        "max_samples": None,
        "batch_size": 15,
        "evaluation_batches": 20,
        "temperature": 0.95,
        "top_p": 0.7,
        "top_k": 50,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.0,
    }


def infer_search(preset_id: str) -> dict:
    if preset_id == "glue_fhx":
        return {
            "popsize": 20,
            "stage1_budget": 400,
            "stage2_budget": 1000,
            "stage1_sigma": 0.05,
            "stage2_sigma": 0.05,
            "scale_bound": 1.5,
            "stage1_l1": 0.05,
            "stage2_l1": 0.05,
        }
    return {
        "popsize": 20,
        "stage1_budget": 400,
        "stage2_budget": 800,
        "stage1_sigma": 0.05,
        "stage2_sigma": 0.05,
        "scale_bound": 1.5,
        "stage1_l1": 0.05,
        "stage2_l1": 0.05,
    }


def sanitize_payload(preset_id: str, payload: dict) -> dict:
    adapter_dict = {
        name: sanitize_path(path)
        for name, path in payload.get("adapter_dict", {}).items()
    }
    base_model_name_or_path = sanitize_path(payload.get("base_model_name_or_path", "models/base-model"))
    sanitized = {
        "preset_id": preset_id,
        "base_model_name_or_path": base_model_name_or_path,
        "adapter_dict": adapter_dict,
        "target_modules": payload.get("target_modules", []),
        "lora_alpha": payload.get("lora_alpha", 16),
        "lora_dropout": payload.get("lora_dropout", 0.1),
        "r": payload.get("r", 8),
        "bias": payload.get("bias", "none"),
        "fan_in_fan_out": payload.get("fan_in_fan_out", False),
        "inference_mode": payload.get("inference_mode", True),
        "init_lora_weights": payload.get("init_lora_weights", True),
        "modules_to_save": payload.get("modules_to_save"),
        "task_type": payload.get("task_type", "CAUSAL_LM"),
        "use_dora": payload.get("use_dora", False),
        "use_rslora": payload.get("use_rslora", False),
        "query_plus": infer_query_plus(preset_id, payload),
        "search": infer_search(preset_id),
        "metadata": {"legacy_source": sanitize_path(str(SOURCES[preset_id]))},
    }
    return sanitized


def main() -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    for preset_id, source in SOURCES.items():
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        sanitized = sanitize_payload(preset_id, payload)
        target = PRESET_DIR / f"{preset_id}.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(sanitized, handle, indent=2, ensure_ascii=False)
        print(f"Wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
