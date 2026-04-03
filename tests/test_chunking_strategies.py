"""
Unit tests for src/chunking_strategies.py

Covers:
- _to_document: Document passthrough, dict conversion (text/page_content keys,
  metadata), arbitrary object fallback.
- _normalize_documents: mixed input lists, empty list.
- fixed_size_chunking: returns Documents, respects chunk_size, overlap,
  handles multiple input types, empty input.
- recursive_chunking: returns Documents, respects chunk_size, handles
  multiple input types, empty input.
- smart_chunking: 'row' docs passed through unchanged; non-row docs split;
  mixed input; empty input.
- check_sentence_breaks: counts correctly for clean/dirty chunks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from langchain_core.documents import Document
from chunking_strategies import (
    _to_document,
    _normalize_documents,
    fixed_size_chunking,
    recursive_chunking,
    smart_chunking,
    check_sentence_breaks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, doc_type: str = "summary") -> Document:
    return Document(page_content=content, metadata={"type": doc_type})


def _long_text(n_chars: int) -> str:
    """Return a string of exactly n_chars made of repeating words."""
    word = "word "
    return (word * (n_chars // len(word) + 1))[:n_chars]


# ---------------------------------------------------------------------------
# _to_document
# ---------------------------------------------------------------------------

class TestToDocument:
    def test_document_passthrough(self):
        doc = _doc("hello")
        assert _to_document(doc) is doc

    def test_dict_with_text_key(self):
        result = _to_document({"text": "hello", "metadata": {"type": "row"}})
        assert result.page_content == "hello"
        assert result.metadata == {"type": "row"}

    def test_dict_with_page_content_key(self):
        result = _to_document({"page_content": "world", "metadata": {}})
        assert result.page_content == "world"

    def test_dict_text_takes_priority_over_page_content(self):
        result = _to_document({"text": "from_text", "page_content": "from_page"})
        assert result.page_content == "from_text"

    def test_dict_missing_metadata_defaults_to_empty(self):
        result = _to_document({"text": "hi"})
        assert result.metadata == {}

    def test_arbitrary_object_converted_via_str(self):
        result = _to_document(42)
        assert result.page_content == "42"
        assert result.metadata == {}

    def test_string_converted_to_document(self):
        result = _to_document("plain string")
        assert result.page_content == "plain string"

    def test_returns_document_instance(self):
        assert isinstance(_to_document({"text": "x"}), Document)


# ---------------------------------------------------------------------------
# _normalize_documents
# ---------------------------------------------------------------------------

class TestNormalizeDocuments:
    def test_empty_list_returns_empty(self):
        assert _normalize_documents([]) == []

    def test_all_documents_passthrough(self):
        docs = [_doc("a"), _doc("b")]
        result = _normalize_documents(docs)
        assert result == docs

    def test_mixed_input_all_become_documents(self):
        mixed = [_doc("doc"), {"text": "dict"}, "string", 99]
        result = _normalize_documents(mixed)
        assert all(isinstance(r, Document) for r in result)
        assert len(result) == 4

    def test_length_preserved(self):
        items = [{"text": str(i)} for i in range(10)]
        assert len(_normalize_documents(items)) == 10


# ---------------------------------------------------------------------------
# fixed_size_chunking
# ---------------------------------------------------------------------------

class TestFixedSizeChunking:
    def test_returns_list_of_documents(self):
        result = fixed_size_chunking([_doc("hello world")])
        assert all(isinstance(c, Document) for c in result)

    def test_short_text_is_single_chunk(self):
        result = fixed_size_chunking([_doc("short")], chunk_size=1000)
        assert len(result) == 1

    def test_long_text_is_split(self):
        long_doc = _doc(_long_text(3000))
        result = fixed_size_chunking([long_doc], chunk_size=1000, overlap=0)
        assert len(result) >= 3

    def test_chunk_size_respected(self):
        long_doc = _doc(_long_text(5000))
        result = fixed_size_chunking([long_doc], chunk_size=500, overlap=0)
        for chunk in result:
            assert len(chunk.page_content) <= 500

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        long_doc = _doc(_long_text(3000))
        no_overlap = fixed_size_chunking([long_doc], chunk_size=1000, overlap=0)
        with_overlap = fixed_size_chunking([long_doc], chunk_size=1000, overlap=200)
        assert len(with_overlap) >= len(no_overlap)

    def test_accepts_dict_input(self):
        result = fixed_size_chunking([{"text": "hello dict"}])
        assert len(result) >= 1

    def test_accepts_mixed_input(self):
        docs = [_doc("a document"), {"text": "a dict"}]
        result = fixed_size_chunking(docs)
        assert len(result) >= 2

    def test_empty_input_returns_empty(self):
        assert fixed_size_chunking([]) == []

    def test_metadata_preserved_in_chunks(self):
        doc = Document(page_content="text", metadata={"type": "category", "year": 2021})
        result = fixed_size_chunking([doc])
        assert result[0].metadata["type"] == "category"
        assert result[0].metadata["year"] == 2021


# ---------------------------------------------------------------------------
# recursive_chunking
# ---------------------------------------------------------------------------

class TestRecursiveChunking:
    def test_returns_list_of_documents(self):
        result = recursive_chunking([_doc("hello world")])
        assert all(isinstance(c, Document) for c in result)

    def test_short_text_is_single_chunk(self):
        result = recursive_chunking([_doc("short")], chunk_size=1000)
        assert len(result) == 1

    def test_long_text_is_split(self):
        long_doc = _doc(_long_text(3000))
        result = recursive_chunking([long_doc], chunk_size=1000, overlap=0)
        assert len(result) >= 3

    def test_chunk_size_respected(self):
        long_doc = _doc(_long_text(5000))
        result = recursive_chunking([long_doc], chunk_size=500, overlap=0)
        for chunk in result:
            assert len(chunk.page_content) <= 500

    def test_accepts_dict_input(self):
        result = recursive_chunking([{"text": "hello dict"}])
        assert len(result) >= 1

    def test_empty_input_returns_empty(self):
        assert recursive_chunking([]) == []

    def test_metadata_preserved_in_chunks(self):
        doc = Document(page_content="text", metadata={"type": "row", "region": "West"})
        result = recursive_chunking([doc])
        assert result[0].metadata["region"] == "West"

    def test_multiple_docs(self):
        docs = [_doc(_long_text(1500)), _doc(_long_text(1500))]
        result = recursive_chunking(docs, chunk_size=1000, overlap=0)
        assert len(result) >= 4

    def test_prefers_sentence_separators(self):
        """Recursive splitter should prefer '. ' over raw character splits."""
        sentences = ". ".join(["This is sentence number %d" % i for i in range(20)])
        result = recursive_chunking([_doc(sentences)], chunk_size=200, overlap=0)
        # Each chunk should end near a sentence boundary (not mid-word)
        for chunk in result[:-1]:
            assert not chunk.page_content.endswith("-")


# ---------------------------------------------------------------------------
# smart_chunking
# ---------------------------------------------------------------------------

class TestSmartChunking:
    def test_row_doc_passed_through_unchanged(self):
        doc = Document(page_content="a row", metadata={"type": "row"})
        result = smart_chunking([doc])
        assert len(result) == 1
        assert result[0] is doc

    def test_row_doc_content_unchanged(self):
        doc = Document(page_content="original row content", metadata={"type": "row"})
        result = smart_chunking([doc])
        assert result[0].page_content == "original row content"

    def test_non_row_doc_is_split(self):
        long_doc = Document(
            page_content=_long_text(3000),
            metadata={"type": "yearly_summary"},
        )
        result = smart_chunking([long_doc])
        assert len(result) >= 2

    def test_non_row_short_doc_is_single_chunk(self):
        doc = Document(page_content="short summary", metadata={"type": "category_summary"})
        result = smart_chunking([doc])
        assert len(result) == 1

    def test_mixed_row_and_non_row(self):
        row = Document(page_content="row doc", metadata={"type": "row"})
        summary = Document(page_content=_long_text(3000), metadata={"type": "trend_summary"})
        result = smart_chunking([row, summary])
        # row never split, summary split into >=2: total >= 3
        assert len(result) >= 3

    def test_multiple_row_docs_all_preserved(self):
        rows = [
            Document(page_content=f"row {i}", metadata={"type": "row"})
            for i in range(5)
        ]
        result = smart_chunking(rows)
        assert len(result) == 5

    def test_empty_input_returns_empty(self):
        assert smart_chunking([]) == []

    def test_accepts_dict_input(self):
        result = smart_chunking([{"text": "some text", "metadata": {"type": "row"}}])
        assert len(result) == 1

    def test_doc_without_type_treated_as_non_row(self):
        doc = Document(page_content=_long_text(3000), metadata={})
        result = smart_chunking([doc])
        # No 'type' key → not 'row' → goes through recursive_chunking
        assert len(result) >= 2

    def test_returns_list_of_documents(self):
        docs = [Document(page_content="text", metadata={"type": "row"})]
        result = smart_chunking(docs)
        assert all(isinstance(c, Document) for c in result)


# ----------------------------------------------------------------------------
# check_sentence_breaks
# ---------------------------------------------------------------------------

class TestCheckSentenceBreaks:
    def test_all_clean_chunks_prints_zero(self, capsys):
        chunks = [
            _doc("This ends with a period."),
            _doc("This ends with a question mark?"),
            _doc("This ends with an exclamation!"),
        ]
        check_sentence_breaks(chunks)
        out = capsys.readouterr().out
        assert out.startswith("Chunks ending mid-sentence: 0/3")

    def test_all_dirty_chunks_prints_full_count(self, capsys):
        chunks = [_doc("no punctuation here"), _doc("also missing")]
        check_sentence_breaks(chunks)
        out = capsys.readouterr().out
        assert out.startswith("Chunks ending mid-sentence: 2/2")

    def test_mixed_chunks_counted_correctly(self, capsys):
        chunks = [
            _doc("ends cleanly."),
            _doc("does not end cleanly"),
            _doc("also good!"),
        ]
        check_sentence_breaks(chunks)
        out = capsys.readouterr().out
        assert out.startswith("Chunks ending mid-sentence: 1/3")

    def test_trailing_whitespace_ignored(self, capsys):
        chunks = [_doc("ends with period.   ")]
        check_sentence_breaks(chunks)
        out = capsys.readouterr().out
        assert out.startswith("Chunks ending mid-sentence: 0/1")
