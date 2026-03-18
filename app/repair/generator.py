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
        """Tạo hướng dẫn sửa lỗi có thứ tự cho các lỗi đã phân loại.

        Args:
            errors: Danh sách lỗi đã phân loại từ pipeline chấm điểm.
            reference_code: Code reference (có thể là inferred reference) để làm bối cảnh.
            student_code: Code của sinh viên.

        Returns:
            A :class:`~app.models.schemas.RepairGuide` với các bước được sắp xếp theo thứ tự.
        """
        if not errors:
            return RepairGuide(
                steps=[],
                summary="Không phát hiện lỗi — bài nộp có vẻ đúng!",
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
        """Xây dựng tóm tắt ngắn gọn về các lỗi tìm thấy.

        Args:
            errors: Danh sách lỗi đã phân loại.

        Returns:
            Chuỗi tóm tắt bằng tiếng Việt.
        """
        critical = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
        major = sum(1 for e in errors if e.severity == ErrorSeverity.MAJOR)
        minor = sum(1 for e in errors if e.severity == ErrorSeverity.MINOR)
        parts: List[str] = []
        if critical:
            parts.append(f"{critical} lỗi nghiêm trọng")
        if major:
            parts.append(f"{major} lỗi quan trọng")
        if minor:
            parts.append(f"{minor} lỗi nhỏ")
        return f"Phát hiện {', '.join(parts)}. Hãy làm theo các bước dưới đây để sửa bài nộp của bạn."
