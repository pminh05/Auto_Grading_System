"""Integration tests for the FastAPI endpoints using httpx AsyncClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_gemini():
    """Patch GeminiClient so tests do not require a real API key."""
    with patch("app.api.routes.GeminiClient") as MockClass:
        instance = MagicMock()
        instance.generate_feedback = AsyncMock(return_value="Mock feedback")
        instance.generate_repair_hint = AsyncMock(return_value="Mock hint")
        instance.generate_summary = AsyncMock(return_value="Mock summary")
        MockClass.return_value = instance
        # Also patch RepairGuideGenerator to use the mock client
        with patch("app.api.routes.RepairGuideGenerator") as MockRepair:
            from app.models.schemas import RepairGuide
            repair_instance = MagicMock()
            repair_instance.generate = AsyncMock(return_value=RepairGuide(steps=[], summary="All good"))
            MockRepair.return_value = repair_instance
            yield instance


async def test_health_endpoint():
    """GET / should return 200 with status ok."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_health_detail_endpoint():
    """GET /health should return 200."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200


async def test_execute_endpoint():
    """POST /api/v1/execute should run code and return results."""
    payload = {
        "code": "print('hello')",
        "test_cases": [{"input": "", "expected_output": "hello"}],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "passed" in data
    assert "failed" in data
    assert data["passed"] == 1
    assert data["failed"] == 0


async def test_execute_endpoint_failing_test():
    """POST /api/v1/execute should correctly report a failing test."""
    payload = {
        "code": "print('wrong')",
        "test_cases": [{"input": "", "expected_output": "correct"}],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["failed"] == 1


async def test_analyze_endpoint():
    """POST /api/v1/analyze should return graph_diff and similarity_score."""
    payload = {
        "reference_code": "x = 1\nfor i in range(5):\n    x += i\n",
        "student_code": "x = 0\n",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "graph_diff" in data
    assert "similarity_score" in data
    assert 0.0 <= data["similarity_score"] <= 1.0


async def test_analyze_syntax_error():
    """POST /api/v1/analyze with invalid code should return 422."""
    payload = {
        "reference_code": "def ok(): pass",
        "student_code": "def broken(:\n    pass",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 422


async def test_grade_endpoint(mock_gemini):
    """POST /api/v1/grade should return a score and feedback."""
    payload = {
        "question": "Write a function that adds two numbers.",
        "reference_solution": "def add(a, b):\n    return a + b\n",
        "student_code": "def add(a, b):\n    return a + b\n",
        "test_cases": [
            {"input": "", "expected_output": ""},
        ],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/grade", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "feedback" in data
    assert "repair_guide" in data
    assert "execution_result" in data
    assert "graph_diff" in data
    assert data["score"]["total_score"] >= 0


async def test_grade_syntax_error():
    """POST /api/v1/grade with broken student code should return 422."""
    payload = {
        "question": "test",
        "reference_solution": "x = 1",
        "student_code": "def broken(:\n    pass",
        "test_cases": [],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/grade", json=payload)
    assert resp.status_code == 422
