"""Repair guide generator: produce ordered, actionable repair steps."""

from __future__ import annotations

import logging
from typing import List

from app.llm.gemini_client import GeminiClient
from app.models.schemas import ClassifiedError, ErrorSeverity, RepairGuide, RepairStep

logger = logging.getLogger(__name__)


class RepairGuideGenerator:
    """Generate a step-by-step repair guide from a list of classified errors.

    Each error is mapped to one or more :class:`~app.models.schemas.RepairStep`
    objects. For each step a natural-language hint is requested from the
    :class:`~app.llm.gemini_client.GeminiClient`.

    Attributes:
        _client: The LLM client used to generate hints.
    """

    # Severity → priority mapping (lower number = higher priority)
    _SEVERITY_PRIORITY: dict = {
        ErrorSeverity.CRITICAL: 1,
        ErrorSeverity.MAJOR: 2,
        ErrorSeverity.MINOR: 3,
    }

    def __init__(self, gemini_client: GeminiClient | None = None) -> None:
        """Initialise with an optional pre-built GeminiClient.

        Args:
            gemini_client: A :class:`~app.llm.gemini_client.GeminiClient`
                instance.  If *None*, a new instance is created automatically.
        """
        self._client: GeminiClient = gemini_client or GeminiClient()

    async def generate(
        self,
        errors: List[ClassifiedError],
        reference_code: str,
        student_code: str,
    ) -> RepairGuide:
        """Generate an ordered repair guide for the given errors.

        Args:
            errors: Classified errors from the grading pipeline.
            reference_code: Full reference solution source code.
            student_code: Full student submission source code.

        Returns:
            A :class:`~app.models.schemas.RepairGuide` with ordered steps.
        """
        if not errors:
            return RepairGuide(
                steps=[],
                summary="No errors detected — the submission looks correct!",
            )

        # Sort errors by severity priority
        sorted_errors = sorted(
            errors,
            key=lambda e: self._SEVERITY_PRIORITY.get(e.severity, 99),
        )

        steps: List[RepairStep] = []
        for idx, error in enumerate(sorted_errors, start=1):
            hint = await self._get_hint(error, reference_code)
            steps.append(
                RepairStep(
                    step_number=idx,
                    description=error.description,
                    code_example=hint,
                    priority=self._SEVERITY_PRIORITY.get(error.severity, 3),
                )
            )

        summary = self._build_summary(errors)

        return RepairGuide(steps=steps, summary=summary)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_hint(self, error: ClassifiedError, reference_code: str) -> str:
        """Request an LLM-generated hint for a single error.

        Args:
            error: The classified error.
            reference_code: The reference solution (used as context).

        Returns:
            A hint string; falls back to the error description on failure.
        """
        try:
            return await self._client.generate_repair_hint(error, reference_code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate LLM hint for error '%s': %s", error.description, exc)
            return error.description

    @staticmethod
    def _build_summary(errors: List[ClassifiedError]) -> str:
        """Build a brief text summary of the errors found.

        Args:
            errors: List of classified errors.

        Returns:
            A summary string.
        """
        critical = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
        major = sum(1 for e in errors if e.severity == ErrorSeverity.MAJOR)
        minor = sum(1 for e in errors if e.severity == ErrorSeverity.MINOR)
        parts: List[str] = []
        if critical:
            parts.append(f"{critical} critical error(s)")
        if major:
            parts.append(f"{major} major error(s)")
        if minor:
            parts.append(f"{minor} minor error(s)")
        return f"Found {', '.join(parts)}. Please follow the steps below to fix your submission."
