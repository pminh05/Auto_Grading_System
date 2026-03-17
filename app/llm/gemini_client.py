"""Gemini 2.0 Flash integration for feedback and repair hint generation."""

from __future__ import annotations

import logging
from typing import List

import google.generativeai as genai

from app.config import settings
from app.models.schemas import ClassifiedError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wrapper around the Google Generative AI SDK for Gemini 2.0 Flash.

    All methods are synchronous but are wrapped in ``async`` coroutines so
    they can be ``await``-ed inside FastAPI route handlers without blocking
    the event loop via ``asyncio.to_thread``.

    Attributes:
        _model: The :class:`google.generativeai.GenerativeModel` instance.
    """

    _MODEL_NAME = "gemini-2.0-flash-preview"
    _FALLBACK_MODEL = "gemini-2.0-flash-exp"

    def __init__(self) -> None:
        """Configure the Gemini SDK using the API key from settings."""
        genai.configure(api_key=settings.gemini_api_key)
        try:
            self._model = genai.GenerativeModel(self._MODEL_NAME)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Model '%s' not available, falling back to '%s'.",
                self._MODEL_NAME,
                self._FALLBACK_MODEL,
            )
            self._model = genai.GenerativeModel(self._FALLBACK_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_feedback(
        self,
        question: str,
        student_code: str,
        errors: List[ClassifiedError],
    ) -> str:
        """Generate natural-language feedback for a student submission.

        Args:
            question: The problem statement.
            student_code: The student's source code.
            errors: List of classified errors from the grading pipeline.

        Returns:
            A multi-sentence feedback string in Vietnamese/English.
        """
        error_descriptions = "\n".join(
            f"- [{e.severity.value}] {e.error_type.value}: {e.description}"
            for e in errors
        ) or "No significant errors found."

        prompt = f"""You are an expert programming tutor. Analyze the student's code submission and provide constructive feedback in Vietnamese.

**Problem Statement:**
{question}

**Student's Code:**
```python
{student_code}
```

**Detected Issues:**
{error_descriptions}

Please provide:
1. A brief assessment of the student's understanding
2. Specific feedback on each detected issue
3. Encouragement and positive reinforcement
4. Overall suggestions for improvement

Keep the tone supportive and educational. Respond in Vietnamese."""

        return await self._generate(prompt)

    async def generate_summary(self, code: str) -> str:
        """Summarise what the student's code does.

        Args:
            code: Python source code to summarise.

        Returns:
            A short summary string.
        """
        prompt = f"""Summarize the following Python code in 2-3 sentences. Describe what it does, its main logic, and any notable patterns. Be concise and clear.

```python
{code}
```

Respond in English."""
        return await self._generate(prompt)

    async def generate_repair_hint(self, error: ClassifiedError, reference_snippet: str) -> str:
        """Generate a specific repair hint for a single error.

        Args:
            error: The classified error to generate a hint for.
            reference_snippet: A snippet from the reference solution illustrating
                the correct approach.

        Returns:
            A step-by-step hint string.
        """
        prompt = f"""You are a programming tutor helping a student fix a bug. Provide a clear, step-by-step hint to fix the following error.

**Error Type:** {error.error_type.value}
**Severity:** {error.severity.value}
**Description:** {error.description}

**Reference Solution Snippet (for context):**
```python
{reference_snippet}
```

Provide:
1. What is wrong
2. How to fix it (step by step)
3. A corrected code example

Keep it educational — guide the student without simply giving the answer. Respond in Vietnamese."""

        return await self._generate(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate(self, prompt: str) -> str:
        """Call the Gemini API and return the response text.

        Falls back to an error message if the API call fails.

        Args:
            prompt: The full prompt string.

        Returns:
            Generated text or an error placeholder.
        """
        import asyncio

        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            return response.text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini API call failed: %s", exc)
            return f"[Feedback generation unavailable: {exc}]"
