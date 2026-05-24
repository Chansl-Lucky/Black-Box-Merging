from __future__ import annotations

import argparse
import json

from BMM.core.runner import run_preset_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BMM preset with optional overrides.")
    parser.add_argument("--preset", required=True, help="Preset id under bmm_service/presets.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--cutoff_len", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--evaluation_batches", type=int, default=None)
    args = parser.parse_args()

    summary = run_preset_optimization(
        preset_id=args.preset,
        model_name_or_path=args.model_name_or_path,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        template=args.template,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        evaluation_batches=args.evaluation_batches,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
