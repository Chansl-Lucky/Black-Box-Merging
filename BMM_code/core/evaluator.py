from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def collate_fn(batch, tokenizer):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


@dataclass
class QueryPlus:
    dataset: str
    dataset_dir: str = "data/Userdata"
    template: str = "default"
    cutoff_len: int = 2048
    max_samples: int | None = None
    batch_size: int = 15
    evaluation_batches: int = 20
    temperature: float = 0.95
    top_p: float = 0.7
    top_k: int = 50
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict) -> "QueryPlus":
        return cls(**payload)

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "dataset_dir": self.dataset_dir,
            "template": self.template,
            "cutoff_len": self.cutoff_len,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "evaluation_batches": self.evaluation_batches,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }


class MergeEvaluator:
    def __init__(self, model_name_or_path: str, workdir: Path):
        self.model_name_or_path = model_name_or_path
        self.workdir = workdir
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._base_model = None
        self._dataloader_cache: dict[tuple, object] = {}

    def get_base_model(self):
        if self._base_model is None:
            from transformers import AutoModelForCausalLM

            self._base_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        return self._base_model

    def build_dataloader(self, query_plus: QueryPlus):
        from torch.utils.data import DataLoader
        from transformers import Seq2SeqTrainingArguments

        from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
        from llamafactory.hparams import get_infer_args
        from llamafactory.model import load_tokenizer

        cache_key = tuple(sorted(query_plus.as_dict().items()))
        if cache_key in self._dataloader_cache:
            return self._dataloader_cache[cache_key]

        model_args, data_args, _, _ = get_infer_args(
            {
                "model_name_or_path": self.model_name_or_path,
                "dataset": query_plus.dataset,
                "dataset_dir": query_plus.dataset_dir,
                "template": query_plus.template,
                "cutoff_len": query_plus.cutoff_len,
                "max_samples": query_plus.max_samples,
                "temperature": query_plus.temperature,
                "top_p": query_plus.top_p,
                "top_k": query_plus.top_k,
                "max_new_tokens": query_plus.max_new_tokens,
                "repetition_penalty": query_plus.repetition_penalty,
            }
        )

        training_args = Seq2SeqTrainingArguments(output_dir=str(self.workdir / "tmp_trainer_output"))
        tokenizer_bundle = load_tokenizer(model_args)
        tokenizer = tokenizer_bundle["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        template.mm_plugin.expand_mm_tokens = False
        dataset_bundle = get_dataset(template, model_args, data_args, training_args, "sft", **tokenizer_bundle)
        train_dataset = dataset_bundle.get("train_dataset") or dataset_bundle.get("eval_dataset")

        dataloader = DataLoader(
            train_dataset,
            batch_size=query_plus.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )
        self._dataloader_cache[cache_key] = dataloader
        return dataloader

    def evaluate(self, adapter_dir: Path, query_plus: QueryPlus) -> float:
        import torch
        from peft import PeftModel

        dataloader = self.build_dataloader(query_plus)
        model = PeftModel.from_pretrained(self.get_base_model(), str(adapter_dir))
        model.to(self.device)
        model.eval()

        total_loss = 0.0
        data_iter = iter(dataloader)
        with torch.no_grad():
            for _ in range(query_plus.evaluation_batches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "labels": batch["labels"].to(self.device),
                }
                outputs = model(**inputs)
                total_loss += float(outputs.loss.item())

        avg_loss = total_loss / max(1, query_plus.evaluation_batches)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return avg_loss
