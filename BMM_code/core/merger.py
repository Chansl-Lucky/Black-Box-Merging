from __future__ import annotations

import json
import shutil
from pathlib import Path

def sparsify_tensor_by_magnitude(tensor, keep_fraction: float):
    import torch

    keep_fraction = max(0.0, min(1.0, float(keep_fraction)))
    if tensor.numel() == 0:
        return tensor.clone()
    if keep_fraction == 0.0:
        return torch.zeros_like(tensor)
    if keep_fraction == 1.0:
        return tensor.clone()

    keep_count = int(round(tensor.numel() * keep_fraction))
    if keep_count <= 0:
        keep_count = 1
    keep_count = min(keep_count, tensor.numel())

    abs_tensor = tensor.abs().reshape(-1)
    threshold = torch.topk(abs_tensor, keep_count).values.min()
    result = tensor.clone()
    result[tensor.abs() < threshold] = 0.0
    return result


def discover_module_keys(adapter_dict: dict[str, Path]) -> list[str]:
    first_adapter_path = next(iter(adapter_dict.values()))
    module_keys: list[str] = []
    from safetensors import safe_open

    with safe_open(first_adapter_path, framework="pt") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight"):
                module_keys.append(key[: -len(".lora_A.weight")])
    return module_keys


def copy_adapter_scaffold(first_adapter_path: Path, output_dir: Path, base_model_name_or_path: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_dir = first_adapter_path.parent
    for name in (
        "adapter_config.json",
        "README.md",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        source = template_dir / name
        if source.exists():
            shutil.copy(source, output_dir / name)

    config_path = output_dir / "adapter_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["base_model_name_or_path"] = base_model_name_or_path
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


def materialize_merged_adapter(
    adapter_dict: dict[str, Path],
    alpha,
    beta,
    output_dir: Path,
    base_model_name_or_path: str,
    preset_id: str,
    query_plus: dict | None = None,
) -> Path:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if len(alpha) != len(adapter_dict):
        raise ValueError(f"alpha length {len(alpha)} does not match adapter count {len(adapter_dict)}")
    if len(beta) != len(adapter_dict):
        raise ValueError(f"beta length {len(beta)} does not match adapter count {len(adapter_dict)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    first_adapter_path = next(iter(adapter_dict.values()))
    copy_adapter_scaffold(first_adapter_path, output_dir, base_model_name_or_path)

    module_keys = discover_module_keys(adapter_dict)
    merged_state: dict[str, torch.Tensor] = {}

    with safe_open(first_adapter_path, framework="pt") as template:
        template_keys = set(template.keys())

    adapter_names = list(adapter_dict.keys())
    for module_key in module_keys:
        lora_a_name = f"{module_key}.lora_A.weight"
        lora_b_name = f"{module_key}.lora_B.weight"
        if lora_a_name not in template_keys or lora_b_name not in template_keys:
            continue

        with safe_open(first_adapter_path, framework="pt") as template:
            fused_a = torch.zeros_like(template.get_tensor(lora_a_name))
            fused_b = torch.zeros_like(template.get_tensor(lora_b_name))

        for index, adapter_name in enumerate(adapter_names):
            adapter_path = adapter_dict[adapter_name]
            with safe_open(adapter_path, framework="pt") as adapter_file:
                if lora_a_name not in adapter_file.keys() or lora_b_name not in adapter_file.keys():
                    continue
                raw_a = adapter_file.get_tensor(lora_a_name)
                raw_b = adapter_file.get_tensor(lora_b_name)
                fused_a += float(beta[index]) * sparsify_tensor_by_magnitude(raw_a, float(alpha[index]))
                fused_b += float(beta[index]) * raw_b

        merged_state[lora_a_name] = fused_a
        merged_state[lora_b_name] = fused_b

    adapter_path = output_dir / "adapter_model.safetensors"
    save_file(merged_state, str(adapter_path))

    alpha_values = alpha.tolist() if hasattr(alpha, "tolist") else list(alpha)
    beta_values = beta.tolist() if hasattr(beta, "tolist") else list(beta)
    manifest = {
        "preset_id": preset_id,
        "alpha": alpha_values,
        "beta": beta_values,
        "query_plus": query_plus or {},
    }
    with (output_dir / "merge_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return adapter_path
