"""
DataLoader: đọc CLC11topic.json và CLC11cv.json, chuẩn hoá sang internal types.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class TopicEntry:
    slug: str
    content: str
    solutions: list[str]   # reference solutions (có thể nhiều)


@dataclass
class SubmissionEntry:
    email: str
    slug: str
    solution: str
    has_runtime_error: bool   # True nếu error != null


class DataLoader:
    """Load và cache topic + submission data từ JSON files."""

    def __init__(
        self,
        topic_path: Path | None = None,
        cv_path: Path | None = None,
    ) -> None:
        self._topic_path = topic_path or DATA_DIR / "CLC11topic.json"
        self._cv_path = cv_path or DATA_DIR / "CLC11cv.json"
        self._topics: dict[str, TopicEntry] | None = None
        self._submissions: list[SubmissionEntry] | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def topics(self) -> dict[str, TopicEntry]:
        if self._topics is None:
            self._topics = self._load_topics()
        return self._topics

    def submissions(self) -> list[SubmissionEntry]:
        if self._submissions is None:
            self._submissions = self._load_submissions()
        return self._submissions

    def submissions_for_slug(self, slug: str) -> list[SubmissionEntry]:
        return [s for s in self.submissions() if s.slug == slug]

    def submissions_for_email(self, email: str) -> list[SubmissionEntry]:
        return [s for s in self.submissions() if s.email == email]

    def unique_emails(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in self.submissions():
            if s.email not in seen:
                seen.add(s.email)
                out.append(s.email)
        return out

    def unique_slugs(self) -> list[str]:
        return list(self.topics().keys())

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_topics(self) -> dict[str, TopicEntry]:
        raw: dict = json.loads(self._topic_path.read_text(encoding="utf-8"))
        return {
            slug: TopicEntry(
                slug=slug,
                content=entry["content"],
                solutions=entry["solutions"],
            )
            for slug, entry in raw.items()
        }

    def _load_submissions(self) -> list[SubmissionEntry]:
        raw: list = json.loads(self._cv_path.read_text(encoding="utf-8"))
        entries: list[SubmissionEntry] = []
        for item in raw:
            entries.append(
                SubmissionEntry(
                    email=item["email"],
                    slug=item["slug"],
                    solution=item["solution"],
                    has_runtime_error=(item.get("error") is not None),
                )
            )
        return entries
