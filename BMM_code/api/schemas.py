from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryPlusModel(BaseModel):
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


class SessionCreateRequest(BaseModel):
    preset_id: str
    model_name_or_path: str | None = None
    query_plus: QueryPlusModel | None = None


class ControlVectorRequest(BaseModel):
    alpha: list[float] = Field(..., description="Sparsity vector.")
    beta: list[float] = Field(..., description="Sign-aware scaling vector.")
    tag: str = "manual"


class OptimizeRequest(BaseModel):
    query_plus: QueryPlusModel | None = None


class SessionResponse(BaseModel):
    session_id: str
    preset_id: str
    workdir: str
    model_name_or_path: str
    query_plus: dict[str, Any]
    adapter_count: int
