"""Tests for document chunkers."""

import pytest

from app.knowledge.documents.chunkers.fixed import FixedChunker
from app.knowledge.documents.chunkers.paragraph import ParagraphChunker
from app.knowledge.documents.chunkers.sentence import SentenceChunker
from app.knowledge.documents.chunkers.recursive import RecursiveChunker
from app.knowledge.models import DocumentFormat, IndexStatus, KnowledgeDocument


def _make_doc(content: str, doc_id: str = "doc1") -> KnowledgeDocument:
    return KnowledgeDocument(
        doc_id=doc_id,
        collection_id="col1",
        namespace="ns1",
        title="Test",
        source_uri="test.txt",
        format=DocumentFormat.TXT,
        content=content,
        content_hash="abc123",
        status=IndexStatus.INDEXED,
    )


# ── FixedChunker ──────────────────────────────────────────────────────


def test_fixed_chunker_basic() -> None:
    chunker = FixedChunker(chunk_size=50, overlap=10)
    doc = _make_doc("A" * 100)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.content) <= 50


def test_fixed_chunker_empty() -> None:
    chunker = FixedChunker(chunk_size=50, overlap=10)
    doc = _make_doc("")
    assert chunker.chunk(doc) == []


def test_fixed_chunker_whitespace_only() -> None:
    chunker = FixedChunker(chunk_size=50, overlap=10)
    doc = _make_doc("   \n\n   ")
    assert chunker.chunk(doc) == []


def test_fixed_chunker_small_text() -> None:
    chunker = FixedChunker(chunk_size=100, overlap=10)
    doc = _make_doc("Hello")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


def test_fixed_chunker_chunk_ids_sequential() -> None:
    chunker = FixedChunker(chunk_size=30, overlap=0)
    doc = _make_doc("A" * 100, doc_id="mydoc")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"mydoc_c{i:04d}"
        assert c.chunk_index == i


def test_fixed_chunker_preserves_metadata() -> None:
    chunker = FixedChunker(chunk_size=50, overlap=10)
    doc = _make_doc("A" * 100)
    doc.metadata["custom"] = "value"
    chunks = chunker.chunk(doc)
    assert all(c.metadata.get("custom") == "value" for c in chunks)


def test_fixed_chunker_overlap_validation() -> None:
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=50, overlap=-1)
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=0, overlap=0)


def test_fixed_chunker_async() -> None:
    import asyncio

    chunker = FixedChunker(chunk_size=50, overlap=10)
    doc = _make_doc("A" * 100)
    chunks = asyncio.run(chunker.chunk_async(doc))
    assert len(chunks) >= 2


# ── ParagraphChunker ──────────────────────────────────────────────────


def test_paragraph_chunker_basic() -> None:
    chunker = ParagraphChunker(chunk_size=100)
    doc = _make_doc("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.content) <= 100


def test_paragraph_chunker_empty() -> None:
    chunker = ParagraphChunker(chunk_size=100)
    doc = _make_doc("")
    assert chunker.chunk(doc) == []


def test_paragraph_chunker_single_paragraph() -> None:
    chunker = ParagraphChunker(chunk_size=200)
    doc = _make_doc("Just one paragraph.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Just one paragraph."


def test_paragraph_chunker_merges_small_paragraphs() -> None:
    chunker = ParagraphChunker(chunk_size=100)
    doc = _make_doc("A.\n\nB.\n\nC.\n\nD.")
    chunks = chunker.chunk(doc)
    # Small paragraphs should be merged
    assert len(chunks) == 1


def test_paragraph_chunker_respects_chunk_size() -> None:
    chunker = ParagraphChunker(chunk_size=30)
    doc = _make_doc("Short.\n\nAnother short.\n\nYet another.")
    chunks = chunker.chunk(doc)
    for c in chunks:
        assert len(c.content) <= 30


def test_paragraph_chunker_custom_separator() -> None:
    chunker = ParagraphChunker(chunk_size=100, separator="\n---\n")
    doc = _make_doc("Part 1\n---\nPart 2\n---\nPart 3")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1


def test_paragraph_chunker_metadata() -> None:
    chunker = ParagraphChunker(chunk_size=100)
    doc = _make_doc("A.\n\nB.")
    chunks = chunker.chunk(doc)
    assert all(c.metadata.get("chunker") == "paragraph" for c in chunks)


# ── SentenceChunker ───────────────────────────────────────────────────


def test_sentence_chunker_basic() -> None:
    chunker = SentenceChunker(chunk_size=100)
    doc = _make_doc("First sentence. Second sentence. Third sentence.")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.content) <= 100


def test_sentence_chunker_empty() -> None:
    chunker = SentenceChunker(chunk_size=100)
    doc = _make_doc("")
    assert chunker.chunk(doc) == []


def test_sentence_chunker_single_sentence() -> None:
    chunker = SentenceChunker(chunk_size=200)
    doc = _make_doc("Just one sentence.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1


def test_sentence_chunker_merges_sentences() -> None:
    chunker = SentenceChunker(chunk_size=100)
    doc = _make_doc("Short. Also short. And this one too.")
    chunks = chunker.chunk(doc)
    # All sentences should fit in one chunk
    assert len(chunks) == 1


def test_sentence_chunker_splits_long() -> None:
    chunker = SentenceChunker(chunk_size=40)
    doc = _make_doc("This is a sentence. Another sentence here. A third one.")
    chunks = chunker.chunk(doc)
    for c in chunks:
        assert len(c.content) <= 40


def test_sentence_chunker_metadata() -> None:
    chunker = SentenceChunker(chunk_size=100)
    doc = _make_doc("A sentence.")
    chunks = chunker.chunk(doc)
    assert all(c.metadata.get("chunker") == "sentence" for c in chunks)


# ── RecursiveChunker ──────────────────────────────────────────────────


def test_recursive_chunker_basic() -> None:
    chunker = RecursiveChunker(chunk_size=50, overlap=0)
    doc = _make_doc("A" * 100)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2


def test_recursive_chunker_empty() -> None:
    chunker = RecursiveChunker(chunk_size=50, overlap=0)
    doc = _make_doc("")
    assert chunker.chunk(doc) == []


def test_recursive_chunker_small_text() -> None:
    chunker = RecursiveChunker(chunk_size=200, overlap=0)
    doc = _make_doc("Hello world.")
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world."


def test_recursive_chunker_uses_paragraphs() -> None:
    chunker = RecursiveChunker(chunk_size=100, overlap=0)
    doc = _make_doc("Para one.\n\nPara two.\n\nPara three.")
    chunks = chunker.chunk(doc)
    # Should split on paragraphs first
    assert len(chunks) >= 1


def test_recursive_chunker_falls_back_to_finer_separators() -> None:
    chunker = RecursiveChunker(chunk_size=30, overlap=0)
    doc = _make_doc("Sentence one. Sentence two. Sentence three. Sentence four.")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.content) <= 60  # Allow some overflow from sentence boundaries


def test_recursive_chunker_custom_separators() -> None:
    chunker = RecursiveChunker(chunk_size=50, overlap=0, separators=("\n", " ", ""))
    doc = _make_doc("Line1\nLine2\nLine3\nLine4")
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1


def test_recursive_chunker_metadata() -> None:
    chunker = RecursiveChunker(chunk_size=50, overlap=0)
    doc = _make_doc("A" * 100)
    chunks = chunker.chunk(doc)
    assert all(c.metadata.get("chunker") == "recursive" for c in chunks)


def test_recursive_chunker_preserves_doc_metadata() -> None:
    chunker = RecursiveChunker(chunk_size=50, overlap=0)
    doc = _make_doc("A" * 100)
    doc.metadata["custom"] = "value"
    chunks = chunker.chunk(doc)
    assert all(c.metadata.get("custom") == "value" for c in chunks)


def test_recursive_chunker_overlap_validation() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=50, overlap=-1)
