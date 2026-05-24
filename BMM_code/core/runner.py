from __future__ import annotations

from .evaluator import QueryPlus
from .registry import load_preset
from .session import MergeSession


def run_preset_optimization(
    preset_id: str,
    model_name_or_path: str | None = None,
    dataset: str | None = None,
    dataset_dir: str | None = None,
    template: str | None = None,
    cutoff_len: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
    evaluation_batches: int | None = None,
) -> dict:
    preset = load_preset(preset_id)
    query_plus_payload = dict(preset.query_plus)
    if dataset is not None:
        query_plus_payload["dataset"] = dataset
    if dataset_dir is not None:
        query_plus_payload["dataset_dir"] = dataset_dir
    if template is not None:
        query_plus_payload["template"] = template
    if cutoff_len is not None:
        query_plus_payload["cutoff_len"] = cutoff_len
    if max_samples is not None:
        query_plus_payload["max_samples"] = max_samples
    if batch_size is not None:
        query_plus_payload["batch_size"] = batch_size
    if evaluation_batches is not None:
        query_plus_payload["evaluation_batches"] = evaluation_batches

    query_plus = QueryPlus.from_dict(query_plus_payload)
    session = MergeSession(preset=preset, query_plus=query_plus, model_name_or_path=model_name_or_path)
    summary = session.optimize()
    summary["session"] = session.describe()
    return summary

