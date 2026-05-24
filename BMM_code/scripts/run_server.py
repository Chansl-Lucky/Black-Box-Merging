from __future__ import annotations

import os

import uvicorn

from BMM.api.app import create_app


def main() -> None:
    app = create_app()
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8010"))
    uvicorn.run(app, host=api_host, port=api_port)


if __name__ == "__main__":
    main()

