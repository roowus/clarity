"""Local HTTP API + web UI server.

`clarity-server` (or `clarity serve`) starts a local server that:
- serves the web UI at /  (static file from clarity/web/index.html)
- POST /analyze {"text": ...} → {"job_id"} (returns immediately)
- GET  /analyze/{job_id} → {"state", "progress", "stage", "result"|"error"}
- GET /api/health → model/device status

Analysis runs in a background worker thread; clients poll for progress so the
UI can show a real progress bar during the ~10-30s inference. Models load ONCE
at startup and stay warm. This is a LOCAL-FIRST tool: bind localhost by
default; --host exposes it deliberately.
"""

from __future__ import annotations

import argparse
import sys
import threading
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from .detector import (
    DEFAULT_THRESHOLD_HIGH,
    DEFAULT_THRESHOLD_LOW,
    FAST_THRESHOLD_HIGH,
    FAST_THRESHOLD_LOW,
    analyze,
)
from .models import ModelPair, FastModel

WEB_DIR = Path(__file__).parent / "web"

# Jobs are small dicts; prune completed ones older than this many entries.
MAX_FINISHED_JOBS = 32


class AnalyzeBody(BaseModel):
    """Module-level so FastAPI's annotation resolver can find it (a
    closure-local class breaks request-body detection and the field degrades
    to a query param)."""

    text: str = Field(min_length=1)
    threshold_low: float | None = None
    threshold_high: float | None = None


def create_app(
    mode: str = "binoculars",
    observer: str | None = None,
    performer: str | None = None,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    scorer: ModelPair | FastModel
    if mode == "fast":
        scorer = FastModel()
    else:
        kwargs = {}
        if observer:
            kwargs["observer_name"] = observer
        if performer:
            kwargs["performer_name"] = performer
        scorer = ModelPair(**kwargs)

    app = FastAPI(title="clarity", version="0.3.0")
    state = {
        "scorer": scorer,
        "mode": mode,
        "jobs": {},  # job_id -> {"state", "progress", "stage", "result"/"error"}
        "order": [],  # completion order for pruning
    }
    # Serialize inference: concurrent MPS/CUDA forward passes contend and can
    # multiply latency (hit during testing). Jobs queue on this lock.
    inference_lock = threading.Lock()

    def _prune_jobs() -> None:
        jobs = state["jobs"]
        while len(state["order"]) > MAX_FINISHED_JOBS:
            old = state["order"].pop(0)
            jobs.pop(old, None)

    def _run_job(job_id: str, body: AnalyzeBody) -> None:
        job = state["jobs"][job_id]

        def progress(pct: int, stage: str) -> None:
            job["progress"], job["stage"] = pct, stage

        try:
            with inference_lock:
                report = analyze(
                    body.text,
                    scorer,
                    threshold_low=body.threshold_low
                    if body.threshold_low is not None
                    else (
                        FAST_THRESHOLD_LOW if state["mode"] == "fast" else DEFAULT_THRESHOLD_LOW
                    ),
                    threshold_high=body.threshold_high
                    if body.threshold_high is not None
                    else (
                        FAST_THRESHOLD_HIGH if state["mode"] == "fast" else DEFAULT_THRESHOLD_HIGH
                    ),
                    progress=progress,
                )
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim to the local client
            job["state"], job["error"] = "error", str(exc)
            return
        job["state"], job["result"] = "done", report.to_dict()
        state["order"].append(job_id)
        _prune_jobs()

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "mode": state["mode"],
            "model": getattr(scorer, "observer_name", None)
            or getattr(scorer, "model_name", None),
            "device": scorer.device,
        }

    @app.post("/analyze")
    def start_analysis(body: AnalyzeBody) -> dict:
        job_id = uuid.uuid4().hex[:12]
        state["jobs"][job_id] = {"state": "running", "progress": 1, "stage": "queued"}
        threading.Thread(target=_run_job, args=(job_id, body), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/analyze/{job_id}")
    def poll(job_id: str) -> dict:
        job = state["jobs"].get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job id")
        out = {"state": job["state"], "progress": job["progress"], "stage": job["stage"]}
        if job["state"] == "done":
            out["result"] = job["result"]
        elif job["state"] == "error":
            out["error"] = job["error"]
        return out

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="clarity-server", description="clarity web UI + API")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default localhost)")
    p.add_argument("--port", type=int, default=8390)
    p.add_argument("--mode", choices=["binoculars", "fast"], default="binoculars")
    p.add_argument("--observer", default=None)
    p.add_argument("--performer", default=None)
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("server extras missing — install with: uv pip install -e '.[serve]'", file=sys.stderr)
        return 2

    import logging

    logging.getLogger("uvicorn.error").info("loading models (first run downloads weights)...")
    app = create_app(mode=args.mode, observer=args.observer, performer=args.performer)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
