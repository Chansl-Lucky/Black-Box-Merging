from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return REPO_ROOT


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def relative_to_repo(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), REPO_ROOT)
    except ValueError:
        return str(path)


def sanitize_path(raw_path: str) -> str:
    path = Path(raw_path)
    parts = path.parts

    if raw_path.startswith(str(REPO_ROOT)):
        return relative_to_repo(Path(raw_path))

    if "adapters" in parts:
        return str(Path(*parts[parts.index("adapters") :]))

    if "src" in parts:
        return str(Path(*parts[parts.index("src") :]))

    if "data" in parts:
        return str(Path(*parts[parts.index("data") :]))

    if "models" in parts:
        return str(Path("models") / path.name)

    return path.name
