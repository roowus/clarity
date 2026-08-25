"""Local HTTP API + web UI server.

`clarity-server` (or `clarity serve`) starts a local server that:
- serves the web UI at /  (static file from clarity/web/index.html)
- POST /analyze {"text": ..., "mode": "binoculars"|"fast"} → Report JSON
- GET /api/health → model/device status

Models load ONCE at startup and stay warm. This is a LOCAL-FIRST tool: bind
localhost by default; --host exposes it deliberately.
"""

from __future__ import annotations

import argparse
import sys
import threading
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

    app = FastAPI(title="clarity", version="0.2.0")
    state = {"scorer": scorer, "mode": mode}
    inference_lock = threading.Lock()

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
    def do_analyze(body: AnalyzeBody) -> dict:
        # Serialize inference: concurrent MPS/CUDA forward passes from the
        # threadpool contend and can multiply latency (hit during testing).
        with inference_lock:
            return _do_analyze_locked(body)

    def _do_analyze_locked(body: AnalyzeBody) -> dict:
        try:
            report = analyze(
                body.text,
                state["scorer"],
                threshold_low=body.threshold_low
                if body.threshold_low is not None
                else (
                    FAST_THRESHOLD_LOW
                    if state["mode"] == "fast"
                    else DEFAULT_THRESHOLD_LOW
                ),
                threshold_high=body.threshold_high
                if body.threshold_high is not None
                else (
                    FAST_THRESHOLD_HIGH
                    if state["mode"] == "fast"
                    else DEFAULT_THRESHOLD_HIGH
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return report.to_dict()

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
