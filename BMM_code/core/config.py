from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import resolve_repo_path


DEFAULT_QUERY_PLUS = {
    "dataset_dir": "data/Userdata",
    "template": "default",
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

DEFAULT_SEARCH = {
    "popsize": 20,
    "stage1_budget": 400,
    "stage2_budget": 800,
    "stage1_sigma": 0.05,
    "stage2_sigma": 0.05,
    "scale_bound": 1.5,
    "stage1_l1": 0.05,
    "stage2_l1": 0.05,
}


@dataclass
class PresetConfig:
    preset_id: str
    base_model_name_or_path: str
    adapter_dict: dict[str, str]
    target_modules: list[str]
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    r: int = 8
    bias: str = "none"
    fan_in_fan_out: bool = False
    inference_mode: bool = True
    init_lora_weights: bool = True
    modules_to_save: list[str] | None = None
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False
    use_rslora: bool = False
    query_plus: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PresetConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        preset_id = payload.get("preset_id", config_path.stem)
        query_plus = dict(DEFAULT_QUERY_PLUS)
        query_plus.update(payload.get("query_plus", {}))

        search = dict(DEFAULT_SEARCH)
        search.update(payload.get("search", {}))

        return cls(
            preset_id=preset_id,
            base_model_name_or_path=payload["base_model_name_or_path"],
            adapter_dict=payload["adapter_dict"],
            target_modules=payload["target_modules"],
            lora_alpha=payload.get("lora_alpha", 16),
            lora_dropout=payload.get("lora_dropout", 0.1),
            r=payload.get("r", 8),
            bias=payload.get("bias", "none"),
            fan_in_fan_out=payload.get("fan_in_fan_out", False),
            inference_mode=payload.get("inference_mode", True),
            init_lora_weights=payload.get("init_lora_weights", True),
            modules_to_save=payload.get("modules_to_save"),
            task_type=payload.get("task_type", "CAUSAL_LM"),
            use_dora=payload.get("use_dora", False),
            use_rslora=payload.get("use_rslora", False),
            query_plus=query_plus,
            search=search,
            metadata=payload.get("metadata", {}),
        )

    def resolved_adapter_dict(self) -> dict[str, Path]:
        return {name: resolve_repo_path(path) for name, path in self.adapter_dict.items()}

    def resolved_base_model_path(self, override: str | None = None) -> str:
        chosen = override or self.base_model_name_or_path
        path = resolve_repo_path(chosen)
        return str(path) if path.exists() else chosen
