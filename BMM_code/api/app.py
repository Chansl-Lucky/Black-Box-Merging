from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - handled at runtime
    FastAPI = None
    HTTPException = RuntimeError

try:
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover - handled at runtime
    CORSMiddleware = None

try:
    from .schemas import ControlVectorRequest, OptimizeRequest, SessionCreateRequest, SessionResponse
except ImportError:  # pragma: no cover - handled at runtime
    ControlVectorRequest = None
    OptimizeRequest = None
    SessionCreateRequest = None
    SessionResponse = None

from ..core.evaluator import QueryPlus
from ..core.registry import list_presets, load_preset
from ..core.session import MergeSession


def create_app() -> FastAPI:
    if FastAPI is None:
        raise RuntimeError("fastapi is required to run the BMM service API.")
    if any(item is None for item in (ControlVectorRequest, OptimizeRequest, SessionCreateRequest, SessionResponse)):
        raise RuntimeError("pydantic is required to run the BMM service API.")

    app = FastAPI(title="BMM Service", version="0.1.0")
    allow_origins = [item.strip() for item in os.getenv("API_ALLOW_ORIGINS", "*").split(",") if item.strip()]
    if CORSMiddleware is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    sessions: dict[str, MergeSession] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/presets")
    def presets():
        return {"items": list_presets()}

    @app.post("/v1/sessions", response_model=SessionResponse)
    def create_session(request: SessionCreateRequest):
        preset = load_preset(request.preset_id)
        query_plus_payload = dict(preset.query_plus)
        if request.query_plus is not None:
            query_plus_payload.update(request.query_plus.model_dump())
        query_plus = QueryPlus.from_dict(query_plus_payload)
        session = MergeSession(
            preset=preset,
            query_plus=query_plus,
            model_name_or_path=request.model_name_or_path,
        )
        sessions[session.session_id] = session
        return SessionResponse(**session.describe())

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session.")
        return SessionResponse(**session.describe())

    @app.post("/v1/sessions/{session_id}/materialize")
    def materialize(session_id: str, request: ControlVectorRequest):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session.")
        adapter_dir = session.materialize(request.alpha, request.beta, tag=request.tag)
        return {"adapter_dir": str(adapter_dir), "session": session.describe()}

    @app.post("/v1/sessions/{session_id}/evaluate")
    def evaluate(session_id: str, request: ControlVectorRequest):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session.")
        return session.evaluate(request.alpha, request.beta, tag=request.tag)

    @app.post("/v1/sessions/{session_id}/optimize")
    def optimize(session_id: str, request: OptimizeRequest):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown session.")
        if request.query_plus is not None:
            session.query_plus = QueryPlus.from_dict(request.query_plus.model_dump())
        return session.optimize()

    return app
