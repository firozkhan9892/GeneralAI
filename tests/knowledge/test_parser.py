"""Tests for the format parser."""

import json

import pytest

from app.knowledge.documents.parser import FormatParser
from app.knowledge.exceptions import KnowledgeUnsupportedFormatError
from app.knowledge.models import DocumentFormat


def test_parser_txt() -> None:
    parser = FormatParser()
    doc = parser.parse(b"Hello", source_uri="test.txt")
    assert doc.format == DocumentFormat.TXT
    assert doc.content == "Hello"


def test_parser_markdown() -> None:
    parser = FormatParser()
    doc = parser.parse(b"# Title\nContent", source_uri="doc.md")
    assert doc.format == DocumentFormat.MARKDOWN
    assert doc.title == "Title"


def test_parser_json() -> None:
    parser = FormatParser()
    data = {"title": "T", "content": "C"}
    doc = parser.parse(json.dumps(data).encode(), source_uri="doc.json")
    assert doc.format == DocumentFormat.JSON
    assert doc.title == "T"


def test_parser_html() -> None:
    parser = FormatParser()
    html = "<html><body><p>Hello</p></body></html>"
    doc = parser.parse(html.encode(), source_uri="page.html")
    assert doc.format == DocumentFormat.HTML


def test_parser_unsupported_extension() -> None:
    parser = FormatParser()
    with pytest.raises(KnowledgeUnsupportedFormatError):
        parser.parse(b"data", source_uri="file.xyz")


def test_parser_detect_format() -> None:
    parser = FormatParser()
    assert parser.detect_format("file.txt") == DocumentFormat.TXT
    assert parser.detect_format("file.md") == DocumentFormat.MARKDOWN
    assert parser.detect_format("file.json") == DocumentFormat.JSON
    assert parser.detect_format("file.pdf") == DocumentFormat.PDF
    assert parser.detect_format("file.html") == DocumentFormat.HTML
    assert parser.detect_format("file.htm") == DocumentFormat.HTML
    assert parser.detect_format("file.xyz") == DocumentFormat.TXT


def test_parser_register_custom_format() -> None:
    from app.knowledge.documents.loaders.text import TextLoader

    parser = FormatParser()
    loader = TextLoader()
    parser.register(".custom", loader)
    doc = parser.parse(b"custom content", source_uri="file.custom")
    assert doc.content == "custom content"


def test_parser_get_loader_for_extension() -> None:
    parser = FormatParser()
    loader = parser.get_loader_for_extension(".txt")
    assert loader.format == DocumentFormat.TXT


def test_parser_get_loader_for_mime() -> None:
    parser = FormatParser()
    loader = parser.get_loader_for_mime("application/json")
    assert loader.format == DocumentFormat.JSON


def test_parser_get_loader_unknown_mime() -> None:
    parser = FormatParser()
    with pytest.raises(KnowledgeUnsupportedFormatError):
        parser.get_loader_for_mime("application/unknown")


def test_parser_async() -> None:
    import asyncio

    parser = FormatParser()
    doc = asyncio.run(parser.parse_async(b"async content", source_uri="async.txt"))
    assert doc.content == "async content"


def test_parser_metadata_passthrough() -> None:
    parser = FormatParser()
    doc = parser.parse(
        b"content",
        source_uri="test.txt",
        metadata={"custom": "value"},
    )
    assert doc.metadata.get("custom") == "value"
