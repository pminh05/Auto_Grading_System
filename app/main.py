"""FastAPI application entrypoint for the Auto Grading System."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="Auto Grading System",
    description=(
        "Automated Code Grading System using CFG comparison, sandbox execution, "
        "and LLM-powered feedback generation."
    ),
    version="1.0.0",
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["health"])
async def root() -> dict:
    """Health-check endpoint."""
    return {"status": "ok", "service": "Auto Grading System"}


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Detailed health-check endpoint."""
    return {"status": "ok", "debug": settings.debug}
