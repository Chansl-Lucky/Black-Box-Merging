from __future__ import annotations

import json
import urllib.request


BASE_URL = "http://127.0.0.1:8010"


def post_json(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    session = post_json(
        "/v1/sessions",
        {
            "preset_id": "mit_movie",
            "query_plus": {
                "dataset": "mit-movie_sample300",
                "dataset_dir": "data/Userdata",
                "template": "llama3",
                "batch_size": 15,
                "evaluation_batches": 20,
            },
        },
    )
    session_id = session["session_id"]

    result = post_json(
        f"/v1/sessions/{session_id}/evaluate",
        {
            "alpha": [1.0] * session["adapter_count"],
            "beta": [1.0 / session["adapter_count"]] * session["adapter_count"],
            "tag": "uniform-baseline",
        },
    )

    print(json.dumps({"session": session, "result": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
