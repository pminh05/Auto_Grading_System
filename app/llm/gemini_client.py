"""Gemini 2.0 Flash integration for feedback and repair hint generation."""

from __future__ import annotations

import ast as _ast
import asyncio
import json
import logging
from typing import List, Optional

import google.generativeai as genai

from app.config import settings
from app.models.schemas import ClassifiedError, ComparisonResult, GradingOutput, RepairGuide

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

    async def infer_reference_solution(self, question: str, student_code: str) -> str:
        """Từ đề bài và code sinh viên, Gemini suy ra hướng làm đúng của sinh viên
        và sinh ra reference solution phù hợp với cách tiếp cận đó.

        Lưu ý: Không tạo ra solution hoàn toàn khác — mà tạo ra phiên bản đúng
        của chính hướng tiếp cận mà sinh viên đang cố làm.

        Args:
            question: Đề bài / nội dung câu hỏi.
            student_code: Code Python của sinh viên (có thể sai).

        Returns:
            Chuỗi Python code hợp lệ là reference solution được suy ra.
        """
        prompt = f"""Bạn là một chuyên gia lập trình Python. Nhiệm vụ của bạn là phân tích code của sinh viên và tạo ra một "reference solution" phù hợp với hướng tiếp cận của sinh viên đó.

**Đề bài:**
{question}

**Code của sinh viên (có thể chứa lỗi):**
```python
{student_code}
```

**Nhiệm vụ:**
1. Phân tích xem sinh viên đang cố gắng làm gì (dù code có thể sai).
2. Từ hướng tiếp cận đó, tạo ra một đoạn code Python ĐÚNG, ngắn gọn, phù hợp với đề bài.
3. Reference solution phải sử dụng cùng phong cách và cấu trúc chính như sinh viên đang cố dùng (ví dụ: nếu sinh viên dùng vòng lặp for, hãy dùng for; nếu dùng numpy, hãy dùng numpy).

**Yêu cầu:**
- Chỉ trả về code Python thuần túy, KHÔNG có giải thích, KHÔNG có markdown, KHÔNG có ```python```.
- Code phải có thể chạy được và cho kết quả đúng với đề bài.
- Nếu không thể suy ra hướng làm, hãy tạo một solution đơn giản và đúng cho đề bài."""

        result = await self._generate(prompt)

        # Clean up potential markdown code fences
        result = result.strip()
        if result.startswith("```python"):
            result = result[len("```python"):].strip()
        elif result.startswith("```"):
            result = result[3:].strip()
        if result.endswith("```"):
            result = result[:-3].strip()

        # Validate that the result is parseable Python
        try:
            _ast.parse(result)
        except SyntaxError:
            logger.warning("infer_reference_solution returned unparseable code; using fallback.")
            result = "# reference solution không thể suy ra\npass\n"

        return result

    async def grade_with_llm(
        self,
        question: str,
        student_code: str,
        inferred_reference: str,
        comparison_result: ComparisonResult,
        errors: List[ClassifiedError],
        repair_guide: RepairGuide,
        execution_errors: List[str],
        runtime_error: Optional[str],
    ) -> GradingOutput:
        """Đưa TẤT CẢ 4 đầu vào vào một prompt duy nhất để Gemini trả về điểm và feedback.

        Bốn đầu vào:
        1. ComparisonResult (graph diff — sự khác biệt cấu trúc)
        2. List[ClassifiedError] (phân loại lỗi + mức độ nghiêm trọng)
        3. RepairGuide (các bước sửa lỗi)
        4. Lỗi runtime của sinh viên (execution errors)

        Args:
            question: Đề bài.
            student_code: Code của sinh viên.
            inferred_reference: Reference solution được suy ra từ hướng làm sinh viên.
            comparison_result: Kết quả so sánh đồ thị.
            errors: Danh sách lỗi đã phân loại.
            repair_guide: Hướng dẫn sửa lỗi.
            execution_errors: Danh sách lỗi runtime khi chạy code.
            runtime_error: Lỗi runtime chính (nếu có).

        Returns:
            :class:`~app.models.schemas.GradingOutput` với điểm, feedback và tóm tắt.
        """
        # Format errors
        error_text = "\n".join(
            f"  - [{e.severity.value}] {e.error_type.value}: {e.description}"
            for e in errors
        ) or "  (Không phát hiện lỗi)"

        # Format repair steps
        repair_text = "\n".join(
            f"  Bước {s.step_number}: {s.description}"
            for s in repair_guide.steps
        ) or "  (Không có bước sửa lỗi)"

        # Format graph diff
        missing_nodes_text = ", ".join(n.get("type", "") for n in comparison_result.missing_nodes) or "Không có"
        extra_nodes_text = ", ".join(n.get("type", "") for n in comparison_result.extra_nodes) or "Không có"
        missing_edges_text = ", ".join(e.get("type", "") for e in comparison_result.missing_edges) or "Không có"
        extra_edges_text = ", ".join(e.get("type", "") for e in comparison_result.extra_edges) or "Không có"

        # Format execution errors
        exec_error_text = "\n".join(f"  - {err}" for err in execution_errors) or "  (Không có lỗi runtime)"

        prompt = f"""Bạn là một giáo viên lập trình Python chuyên nghiệp đang chấm điểm bài làm của sinh viên.
Hãy đánh giá bài làm dựa trên 4 nguồn thông tin sau và trả về kết quả theo định dạng JSON.

===== ĐỀ BÀI =====
{question}

===== CODE SINH VIÊN =====
```python
{student_code}
```

===== REFERENCE SOLUTION (suy ra từ hướng làm của sinh viên) =====
```python
{inferred_reference}
```

===== (1) SO SÁNH ĐỒ THỊ CẤU TRÚC (Graph Diff) =====
- Điểm tương đồng đồ thị: {comparison_result.similarity_score:.2f} / 1.0
- Node thiếu so với reference: {missing_nodes_text}
- Node thừa so với reference: {extra_nodes_text}
- Cạnh điều khiển thiếu: {missing_edges_text}
- Cạnh điều khiển thừa: {extra_edges_text}

===== (2) PHÂN LOẠI LỖI (Error Classification) =====
{error_text}

===== (3) HƯỚNG DẪN SỬA LỖI (Repair Steps) =====
{repair_text}

===== (4) LỖI RUNTIME CỦA SINH VIÊN =====
{exec_error_text}

===== YÊU CẦU =====
Dựa vào 4 nguồn thông tin trên, hãy:
1. Đánh giá điểm số trên thang 10 (0.0 đến 10.0).
2. Viết nhận xét chi tiết về bài làm của sinh viên bằng tiếng Việt.
3. Liệt kê các bước sửa lỗi cụ thể (nếu có).
4. Tóm tắt ngắn gọn (1-2 câu) về lỗi chính của sinh viên.

**Hướng dẫn chấm điểm:**
- 9-10: Code đúng hoàn toàn, cấu trúc đồ thị giống reference, không có lỗi runtime
- 7-8: Code đúng phần lớn, có lỗi nhỏ, cấu trúc đồ thị tương đối giống
- 5-6: Code đúng một phần, có lỗi logic hoặc thuật toán đáng kể
- 3-4: Code có nhiều lỗi, cấu trúc đồ thị khác biệt nhiều
- 0-2: Code sai hoàn toàn, không chạy được, hoặc không giải đúng đề bài

Trả về ĐÚNG định dạng JSON sau (không có thêm bất kỳ text nào khác):
{{
  "score": <float 0.0-10.0>,
  "feedback": "<nhận xét chi tiết bằng tiếng Việt>",
  "repair_steps": ["<bước 1>", "<bước 2>", ...],
  "summary": "<tóm tắt 1-2 câu>"
}}"""

        raw = await self._generate(prompt)

        # Robust JSON parsing with fallback
        try:
            # Try to extract JSON from response (handle cases with extra text)
            raw_stripped = raw.strip()
            # Remove markdown code fences if present
            try:
                if "```json" in raw_stripped:
                    raw_stripped = raw_stripped.split("```json")[1].split("```")[0].strip()
                elif raw_stripped.startswith("```") and "```" in raw_stripped[3:]:
                    raw_stripped = raw_stripped[3:].split("```")[0].strip()
            except IndexError:
                pass  # Keep raw_stripped as-is if fence parsing fails

            data = json.loads(raw_stripped)
            score = float(data.get("score", 5.0))
            score = max(0.0, min(10.0, score))
            return GradingOutput(
                score=score,
                feedback=str(data.get("feedback", "")),
                repair_steps=[str(s) for s in data.get("repair_steps", [])],
                summary=str(data.get("summary", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("grade_with_llm: JSON parse failed (%s); using fallback.", exc)
            # Fallback: extract score from text if possible, else use midpoint
            fallback_score = 5.0
            return GradingOutput(
                score=fallback_score,
                feedback=raw if raw and not raw.startswith("[") else "Không thể tạo nhận xét tự động.",
                repair_steps=[],
                summary="Hệ thống không thể phân tích chi tiết bài làm.",
            )

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

        prompt = f"""Bạn là một gia sư lập trình chuyên nghiệp. Hãy phân tích bài nộp của sinh viên và cung cấp phản hồi mang tính xây dựng bằng tiếng Việt.

**Đề bài:**
{question}

**Code của sinh viên:**
```python
{student_code}
```

**Các lỗi phát hiện được:**
{error_descriptions}

Hãy cung cấp:
1. Đánh giá ngắn gọn về mức độ hiểu bài của sinh viên
2. Phản hồi cụ thể về từng lỗi phát hiện
3. Lời khuyến khích và động viên tích cực
4. Gợi ý cải thiện tổng thể

Giữ giọng văn hỗ trợ và mang tính giáo dục. Trả lời bằng tiếng Việt."""

        return await self._generate(prompt)

    async def generate_summary(self, code: str) -> str:
        """Summarise what the student's code does.

        Args:
            code: Python source code to summarise.

        Returns:
            A short summary string.
        """
        prompt = f"""Tóm tắt đoạn code Python sau trong 2-3 câu. Mô tả code làm gì, logic chính, và các pattern đáng chú ý. Ngắn gọn và rõ ràng.

```python
{code}
```

Trả lời bằng tiếng Việt."""
        return await self._generate(prompt)

    async def generate_repair_hint(self, error: ClassifiedError, reference_snippet: str) -> str:
        """Generate a specific repair hint for a single error.

        Args:
            error: The classified error to generate a hint for.
            reference_snippet: A snippet from the reference solution illustrating
                the correct approach.

        Returns:
            A step-by-step hint string in Vietnamese.
        """
        prompt = f"""Bạn là gia sư lập trình đang giúp sinh viên sửa lỗi. Hãy cung cấp gợi ý rõ ràng, từng bước để sửa lỗi sau.

**Loại lỗi:** {error.error_type.value}
**Mức độ:** {error.severity.value}
**Mô tả:** {error.description}

**Đoạn code tham khảo (để làm bối cảnh):**
```python
{reference_snippet}
```

Hãy cung cấp:
1. Vấn đề là gì
2. Cách sửa (từng bước)
3. Ví dụ code đã sửa

Hướng dẫn sinh viên mà không đưa thẳng đáp án. Trả lời bằng tiếng Việt."""

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
        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            return response.text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini API call failed: %s", exc)
            return f"[Feedback generation unavailable: {exc}]"
