"""FastAPI application entry point for the AI code detection service."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.routes import router
from src.api.auth_routes import router as auth_router
from src.auth.store import init_db

app = FastAPI(
    title="AI-Generated Code Detector",
    description=(
        "Detects AI-generated code in competitive programming submissions using "
        "an ensemble of stylometric analysis, CodeBERT classification, LLM-as-judge, "
        "and behavioral signals."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="", tags=["detection"])
app.include_router(auth_router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return RedirectResponse(url="/ui/")


@app.on_event("startup")
async def startup():
    from loguru import logger
    init_db()
    logger.info("AI Code Detector API starting up")
    # Kick off background download of CodeBERT MLM model so it is ready
    # before the first /analyze request arrives.
    try:
        from src.models.codebert_classifier import warm_up_zero_shot
        warm_up_zero_shot()
    except Exception as exc:
        logger.warning(f"Could not start CodeBERT warm-up: {exc}")


if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)
