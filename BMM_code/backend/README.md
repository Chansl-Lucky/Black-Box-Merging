# Backend

This directory is the backend entry layer for the standalone BMM open-source package.

## Start

```bash
cd /path/to/BMM
python3 backend/run_server.py
```

## API

- `GET /health`
- `GET /v1/presets`
- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/materialize`
- `POST /v1/sessions/{session_id}/evaluate`
- `POST /v1/sessions/{session_id}/optimize`
