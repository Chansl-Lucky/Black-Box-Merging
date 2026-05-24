from __future__ import annotations

from pathlib import Path

from .config import PresetConfig
from .paths import repo_root


PRESET_DIR = repo_root() / "presets"


def list_presets() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted(PRESET_DIR.glob("*.json")):
        config = PresetConfig.from_file(path)
        items.append(
            {
                "preset_id": config.preset_id,
                "path": str(path.relative_to(repo_root())),
                "base_model_name_or_path": config.base_model_name_or_path,
            }
        )
    return items


def preset_path(preset_id: str) -> Path:
    return PRESET_DIR / f"{preset_id}.json"


def load_preset(preset_id: str) -> PresetConfig:
    path = preset_path(preset_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown preset: {preset_id}")
    return PresetConfig.from_file(path)
