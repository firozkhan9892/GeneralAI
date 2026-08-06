"""Tests for document loaders."""

import json

import pytest

from app.knowledge.documents.loaders.text import TextLoader
from app.knowledge.documents.loaders.markdown import MarkdownLoader
from app.knowledge.documents.loaders.json_loader import JsonLoader
from app.knowledge.documents.loaders.html import HtmlLoader
from app.knowledge.exceptions import KnowledgeIngestionError
from app.knowledge.models import DocumentFormat, IndexStatus


# ── TextLoader ────────────────────────────────────────────────────────


def test_text_loader_basic() -> None:
    loader = TextLoader()
    doc = loader.load(b"Hello, world!", source_uri="test.txt")
    assert doc.format == DocumentFormat.TXT
    assert doc.content == "Hello, world!"
    assert doc.source_uri == "test.txt"
    assert doc.status == IndexStatus.PENDING
    assert doc.content_hash != ""
    assert doc.doc_id  # non-empty


def test_text_loader_empty_content() -> None:
    loader = TextLoader()
    doc = loader.load(b"", source_uri="empty.txt")
    assert doc.content == ""


def test_text_loader_unicode() -> None:
    loader = TextLoader()
    doc = loader.load("café résumé".encode("utf-8"), source_uri="fr.txt")
    assert "café" in doc.content


def test_text_loader_bad_encoding() -> None:
    loader = TextLoader()
    with pytest.raises(KnowledgeIngestionError):
        loader.load(b"\xff\xfe\x00\x01", source_uri="bad.txt")


def test_text_loader_metadata_passthrough() -> None:
    loader = TextLoader()
    doc = loader.load(
        b"content",
        source_uri="test.txt",
        metadata={"custom_key": "custom_value"},
    )
    assert doc.metadata["custom_key"] == "custom_value"
    assert doc.metadata["format"] == "txt"


def test_text_loader_async() -> None:
    import asyncio

    loader = TextLoader()
    doc = asyncio.run(loader.load_async(b"async content", source_uri="async.txt"))
    assert doc.content == "async content"


# ── MarkdownLoader ────────────────────────────────────────────────────


def test_markdown_loader_basic() -> None:
    loader = MarkdownLoader()
    doc = loader.load(
        b"# Title\n\nSome content here.",
        source_uri="readme.md",
    )
    assert doc.format == DocumentFormat.MARKDOWN
    assert doc.title == "Title"
    assert "# Title" in doc.content


def test_markdown_loader_headings_extracted() -> None:
    loader = MarkdownLoader()
    content = "# H1\n## H2\n### H3\nSome text"
    doc = loader.load(content.encode("utf-8"), source_uri="doc.md")
    headings = doc.metadata.get("headings", [])
    assert len(headings) == 3
    assert headings[0]["level"] == 1
    assert headings[0]["text"] == "H1"


def test_markdown_loader_no_headings() -> None:
    loader = MarkdownLoader()
    doc = loader.load(b"Plain text without headings.", source_uri="plain.md")
    assert doc.title == "Plain text without headings."


def test_markdown_loader_bad_encoding() -> None:
    loader = MarkdownLoader()
    with pytest.raises(KnowledgeIngestionError):
        loader.load(b"\xff\xfe", source_uri="bad.md")


# ── JsonLoader ────────────────────────────────────────────────────────


def test_json_loader_basic() -> None:
    loader = JsonLoader()
    data = {"title": "Doc Title", "content": "Doc content here."}
    doc = loader.load(json.dumps(data).encode("utf-8"), source_uri="doc.json")
    assert doc.format == DocumentFormat.JSON
    assert doc.title == "Doc Title"
    assert doc.content == "Doc content here."


def test_json_loader_custom_text_field() -> None:
    loader = JsonLoader(text_field="body.text")
    data = {"body": {"text": "Nested content"}}
    doc = loader.load(json.dumps(data).encode("utf-8"), source_uri="doc.json")
    assert doc.content == "Nested content"


def test_json_loader_no_text_field() -> None:
    loader = JsonLoader(text_field=None)
    data = {"key": "value"}
    doc = loader.load(json.dumps(data).encode("utf-8"), source_uri="doc.json")
    # Falls back to serialised JSON
    assert "key" in doc.content


def test_json_loader_string_value() -> None:
    loader = JsonLoader(text_field=None)
    doc = loader.load(
        json.dumps("just a string").encode("utf-8"), source_uri="doc.json"
    )
    assert doc.content == "just a string"


def test_json_loader_invalid_json() -> None:
    loader = JsonLoader()
    with pytest.raises(KnowledgeIngestionError):
        loader.load(b"not json at all", source_uri="bad.json")


def test_json_loader_bad_encoding() -> None:
    loader = JsonLoader()
    with pytest.raises(KnowledgeIngestionError):
        loader.load(b"\xff\xfe", source_uri="bad.json")


def test_json_loader_nested_title() -> None:
    loader = JsonLoader(title_field="meta.title")
    data = {"meta": {"title": "Nested Title"}, "content": "body"}
    doc = loader.load(json.dumps(data).encode("utf-8"), source_uri="doc.json")
    assert doc.title == "Nested Title"


def test_json_loader_missing_title_falls_back() -> None:
    loader = JsonLoader(text_field="content", title_field="nonexistent")
    data = {"content": "body text here"}
    doc = loader.load(json.dumps(data).encode("utf-8"), source_uri="doc.json")
    assert doc.title == "body text here"  # fallback to extract_title_from_content


# ── HtmlLoader ────────────────────────────────────────────────────────


def test_html_loader_basic() -> None:
    loader = HtmlLoader()
    html = "<html><body><p>Hello world</p></body></html>"
    doc = loader.load(html.encode("utf-8"), source_uri="page.html")
    assert doc.format == DocumentFormat.HTML
    assert "Hello world" in doc.content


def test_html_loader_strips_scripts() -> None:
    loader = HtmlLoader()
    html = "<html><body><script>alert('x')</script><p>Real content</p></body></html>"
    doc = loader.load(html.encode("utf-8"), source_uri="page.html")
    assert "alert" not in doc.content
    assert "Real content" in doc.content


def test_html_loader_strips_styles() -> None:
    loader = HtmlLoader()
    html = "<html><body><style>.red{color:red}</style><p>Content</p></body></html>"
    doc = loader.load(html.encode("utf-8"), source_uri="page.html")
    assert ".red" not in doc.content


def test_html_loader_extracts_title() -> None:
    loader = HtmlLoader()
    html = "<html><head><title>My Page</title></head><body><p>Content</p></body></html>"
    doc = loader.load(html.encode("utf-8"), source_uri="page.html")
    assert doc.title == "My Page"


def test_html_loader_extracts_h1() -> None:
    loader = HtmlLoader()
    html = "<html><body><h1>Page Heading</h1><p>Content</p></body></html>"
    doc = loader.load(html.encode("utf-8"), source_uri="page.html")
    assert doc.title == "Page Heading"


def test_html_loader_og_title() -> None:
    loader = HtmlLoader()
    html = '<html><head><meta property="og:title" content="OG Title"></head><body><p>Content</p></body></html>'
    doc = loader.load(html.encode("utf-8"), source_uri="page.html")
    assert doc.title == "OG Title"


def test_html_loader_bad_encoding() -> None:
    loader = HtmlLoader()
    with pytest.raises(KnowledgeIngestionError):
        loader.load(b"\xff\xfe", source_uri="bad.html")


def test_html_loader_empty_body() -> None:
    loader = HtmlLoader()
    doc = loader.load(b"<html><body></body></html>", source_uri="empty.html")
    assert doc.content == ""


# ── PdfLoader (mock) ─────────────────────────────────────────────────


def test_pdf_loader_requires_pypdf(monkeypatch: object) -> None:
    """PDF loader raises clear error when pypdf is unavailable."""
    import sys

    # Temporarily remove pypdf from sys.modules
    saved = sys.modules.pop("pypdf", None)
    sys.modules["pypdf"] = None  # type: ignore[assignment]

    try:
        from app.knowledge.documents.loaders.pdf import PdfLoader

        loader = PdfLoader()
        with pytest.raises(KnowledgeIngestionError, match="pypdf is required"):
            loader.load(b"fake pdf", source_uri="doc.pdf")
    finally:
        if saved is not None:
            sys.modules["pypdf"] = saved
        else:
            sys.modules.pop("pypdf", None)
