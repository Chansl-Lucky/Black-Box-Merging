from __future__ import annotations

import json
import uuid
from pathlib import Path

from .config import PresetConfig
from .evaluator import MergeEvaluator, QueryPlus
from .merger import materialize_merged_adapter
from .paths import repo_root


class MergeSession:
    def __init__(
        self,
        preset: PresetConfig,
        query_plus: QueryPlus,
        model_name_or_path: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.preset = preset
        self.query_plus = query_plus
        self.model_name_or_path = preset.resolved_base_model_path(model_name_or_path)
        self.session_id = session_id or uuid.uuid4().hex
        self.workdir = repo_root() / "artifacts" / self.session_id
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.workdir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._evaluator: MergeEvaluator | None = None
        self.adapter_dict = preset.resolved_adapter_dict()

    def get_evaluator(self) -> MergeEvaluator:
        if self._evaluator is None:
            self._evaluator = MergeEvaluator(self.model_name_or_path, self.workdir)
        return self._evaluator

    def describe(self) -> dict:
        return {
            "session_id": self.session_id,
            "preset_id": self.preset.preset_id,
            "workdir": str(self.workdir.relative_to(repo_root())),
            "model_name_or_path": self.model_name_or_path,
            "query_plus": self.query_plus.as_dict(),
            "adapter_count": len(self.adapter_dict),
        }

    def materialize(self, alpha: list[float], beta: list[float], tag: str = "merged") -> Path:
        import numpy as np

        artifact_dir = self.workdir / "artifacts" / tag
        materialize_merged_adapter(
            adapter_dict=self.adapter_dict,
            alpha=np.asarray(alpha, dtype=np.float32),
            beta=np.asarray(beta, dtype=np.float32),
            output_dir=artifact_dir,
            base_model_name_or_path=self.model_name_or_path,
            preset_id=self.preset.preset_id,
            query_plus=self.query_plus.as_dict(),
        )
        return artifact_dir

    def evaluate(self, alpha: list[float], beta: list[float], tag: str = "active") -> dict:
        artifact_dir = self.materialize(alpha, beta, tag=tag)
        loss = self.get_evaluator().evaluate(artifact_dir, self.query_plus)
        return {"loss": loss, "adapter_dir": str(artifact_dir.relative_to(repo_root()))}

    def optimize(self) -> dict:
        import cma
        import numpy as np

        search = self.preset.search
        adapter_count = len(self.adapter_dict)
        uniform_beta = np.full(adapter_count, 1.0 / adapter_count, dtype=np.float32)

        best_loss = float("inf")
        best_alpha = np.ones(adapter_count, dtype=np.float32)
        best_beta = uniform_beta.copy()

        stage1 = cma.CMAEvolutionStrategy(
            adapter_count * [1.0],
            search["stage1_sigma"],
            inopts={
                "seed": 42,
                "popsize": search["popsize"],
                "maxiter": search["stage1_budget"] // search["popsize"],
                "bounds": [0.0, 1.0],
                "verbose": -9,
            },
        )

        stage1_eval_count = 0
        while not stage1.stop() and stage1_eval_count < search["stage1_budget"]:
            solutions = stage1.ask()
            losses = []
            for solution in solutions:
                alpha = np.clip(np.asarray(solution, dtype=np.float32), 0.0, 1.0)
                metrics = self.evaluate(alpha.tolist(), uniform_beta.tolist(), tag="stage1_active")
                loss = float(metrics["loss"]) + search["stage1_l1"] * float(np.mean(np.abs(alpha)))
                losses.append(loss)
                stage1_eval_count += 1
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha.copy()
                    np.save(self.checkpoint_dir / "best_stage1_alpha.npy", best_alpha)
                    self.materialize(best_alpha.tolist(), uniform_beta.tolist(), tag="best_stage1")
            stage1.tell(solutions, losses)

        stage2 = cma.CMAEvolutionStrategy(
            uniform_beta.tolist(),
            search["stage2_sigma"],
            inopts={
                "seed": 42,
                "popsize": search["popsize"],
                "maxiter": search["stage2_budget"] // search["popsize"],
                "bounds": [-search["scale_bound"], search["scale_bound"]],
                "verbose": -9,
            },
        )

        stage2_eval_count = 0
        while not stage2.stop() and stage2_eval_count < search["stage2_budget"]:
            solutions = stage2.ask()
            losses = []
            for solution in solutions:
                beta = np.clip(
                    np.asarray(solution, dtype=np.float32),
                    -search["scale_bound"],
                    search["scale_bound"],
                )
                metrics = self.evaluate(best_alpha.tolist(), beta.tolist(), tag="stage2_active")
                loss = float(metrics["loss"]) + search["stage2_l1"] * float(np.mean(np.abs(beta)))
                losses.append(loss)
                stage2_eval_count += 1
                if loss < best_loss:
                    best_loss = loss
                    best_beta = beta.copy()
                    np.save(self.checkpoint_dir / "best_stage2_beta.npy", best_beta)
                    self.materialize(best_alpha.tolist(), best_beta.tolist(), tag="best_stage2")
            stage2.tell(solutions, losses)

        summary = {
            "best_loss": best_loss,
            "best_alpha": best_alpha.tolist(),
            "best_beta": best_beta.tolist(),
            "checkpoint_dir": str(self.checkpoint_dir.relative_to(repo_root())),
            "stage1_evaluations": stage1_eval_count,
            "stage2_evaluations": stage2_eval_count,
        }
        final_best_dir = self.materialize(best_alpha.tolist(), best_beta.tolist(), tag="final_best")
        summary["best_adapter_dir"] = str(final_best_dir.relative_to(repo_root()))
        with (self.workdir / "optimization_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary
