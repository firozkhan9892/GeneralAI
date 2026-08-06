"""Tests for knowledge utility functions."""

from app.knowledge.utils import (
    compute_content_hash,
    estimate_token_count,
    extract_title_from_content,
    normalise_whitespace,
)


def test_compute_content_hash_deterministic() -> None:
    h1 = compute_content_hash("Hello, world!")
    h2 = compute_content_hash("Hello, world!")
    assert h1 == h2


def test_compute_content_hash_differs_for_different_content() -> None:
    h1 = compute_content_hash("Alpha")
    h2 = compute_content_hash("Beta")
    assert h1 != h2


def test_compute_content_hash_strips_whitespace() -> None:
    h1 = compute_content_hash("  Hello  ")
    h2 = compute_content_hash("Hello")
    assert h1 == h2


def test_compute_content_hash_empty() -> None:
    h = compute_content_hash("")
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256 hex digest


def test_estimate_token_count_positive() -> None:
    assert estimate_token_count("Hello, world!") >= 1


def test_estimate_token_count_empty() -> None:
    assert estimate_token_count("") >= 1


def test_estimate_token_count_scales_with_length() -> None:
    short = estimate_token_count("Hi")
    long = estimate_token_count("A" * 1000)
    assert long > short


def test_normalise_whitespace() -> None:
    assert normalise_whitespace("  hello   world  ") == "hello world"


def test_normalise_whitespace_newlines() -> None:
    assert normalise_whitespace("hello\n\nworld") == "hello world"


def test_extract_title_from_content_with_heading() -> None:
    text = "# My Title\nSome content here"
    title = extract_title_from_content(text)
    assert title == "# My Title"


def test_extract_title_from_content_with_bare_line() -> None:
    text = "First line is the title\nMore content"
    title = extract_title_from_content(text)
    assert title == "First line is the title"


def test_extract_title_from_content_empty() -> None:
    assert extract_title_from_content("") == ""


def test_extract_title_from_content_max_len() -> None:
    text = "A" * 200
    title = extract_title_from_content(text, max_len=50)
    assert len(title) == 50
