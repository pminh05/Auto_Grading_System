"""Tests for DataLoader."""
import pytest
from pathlib import Path
from app.data_loader import DataLoader, TopicEntry, SubmissionEntry


def test_topics_loaded():
    loader = DataLoader()
    topics = loader.topics()
    assert len(topics) == 4
    for slug, topic in topics.items():
        assert slug == topic.slug
        assert len(topic.solutions) >= 1
        assert topic.content


def test_submissions_loaded():
    loader = DataLoader()
    subs = loader.submissions()
    assert len(subs) > 0
    for s in subs:
        assert s.email
        assert s.slug
        assert s.solution


def test_submissions_for_slug():
    loader = DataLoader()
    slug = "numpy-01-phan-tich-loi-nhuan-ban-hang-68dbc899fdd70279dba7dd24"
    subs = loader.submissions_for_slug(slug)
    assert len(subs) > 0
    assert all(s.slug == slug for s in subs)


def test_runtime_error_flag():
    loader = DataLoader()
    subs = loader.submissions()
    # có ít nhất 1 bài lỗi và 1 bài không lỗi
    assert any(s.has_runtime_error for s in subs)
    assert any(not s.has_runtime_error for s in subs)


def test_unique_emails():
    loader = DataLoader()
    emails = loader.unique_emails()
    assert len(emails) == len(set(emails))  # không trùng lặp
    assert len(emails) > 0


def test_submissions_for_email():
    loader = DataLoader()
    emails = loader.unique_emails()
    email = emails[0]
    subs = loader.submissions_for_email(email)
    assert len(subs) > 0
    assert all(s.email == email for s in subs)
