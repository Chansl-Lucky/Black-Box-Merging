from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".py", ".json", ".md", ".sh", ".toml"}

REPLACEMENTS = [
    ("", ""),
    ("", ""),
    ("../", "../"),
    ("../", "../"),
    ("models/", "models/"),
    ("models/", "models/"),
]


def should_process(path: Path) -> bool:
    if path.suffix not in TEXT_SUFFIXES:
        return False
    parts = set(path.parts)
    if ".git" in parts or "__pycache__" in parts:
        return False
    return path.is_file()


def sanitize_file(path: Path) -> bool:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False

    updated = original
    for old, new in REPLACEMENTS:
        updated = updated.replace(old, new)

    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    changed = []
    for path in ROOT.rglob("*"):
        if should_process(path) and sanitize_file(path):
            changed.append(path.relative_to(ROOT))

    for path in changed:
        print(path)
    print(f"changed_files={len(changed)}")


if __name__ == "__main__":
    main()
