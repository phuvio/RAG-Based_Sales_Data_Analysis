"""
Unit tests for src/build_index.py

Strategy: patch module-level globals (client, embedding_model) and the
generate_all_documents import so no real ChromaDB, network, or GPU is needed.

Covers:
- Documents are built from generate_all_documents output.
- Embeddings are produced for every document text.
- Old collection is deleted before creating a new one.
- Documents are batched correctly (all added, respecting batch_size).
- The new collection is returned.
- Correct IDs (doc-0, doc-1, …) are assigned.
- Edge cases: empty dataframe, exactly batch_size docs, multiple batches.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_docs(n: int) -> list[dict]:
    """Return n minimal raw doc dicts matching generate_documents output."""
    return [
        {"text": f"doc text {i}", "metadata": {"type": "row", "index": i}}
        for i in range(n)
    ]


def _fake_encode(texts, show_progress_bar=False):
    """Return a float32 ndarray with one 4-dim embedding per text."""
    return np.array([[float(i)] * 4 for i in range(len(texts))], dtype="float32")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_globals():
    """
    Patch all module-level globals in build_index before every test so tests
    are fully isolated from external services.
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_collection

    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.side_effect = _fake_encode

    with (
        patch("build_index.client", mock_client),
        patch("build_index.embedding_model", mock_embedding_model),
        patch("build_index.generate_all_documents") as mock_gen,
    ):
        yield {
            "client": mock_client,
            "collection": mock_collection,
            "embedding_model": mock_embedding_model,
            "generate_all_documents": mock_gen,
        }


# ---------------------------------------------------------------------------
# Import the module AFTER patching path is set up.
# ---------------------------------------------------------------------------

import build_index  # noqa: E402  (must follow sys.path insert)


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

class TestDocumentGeneration:
    def test_generate_all_documents_called_with_df(self, patch_globals):
        df = MagicMock()
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(3)
        build_index.build_index(df)
        patch_globals["generate_all_documents"].assert_called_once_with(df)

    def test_empty_raw_docs_produces_no_embeddings(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = []
        build_index.build_index(MagicMock())
        patch_globals["embedding_model"].encode.assert_called_once_with([], show_progress_bar=True)

    def test_texts_extracted_from_raw_docs(self, patch_globals):
        raw = _make_raw_docs(3)
        patch_globals["generate_all_documents"].return_value = raw
        build_index.build_index(MagicMock())
        encode_call_texts = patch_globals["embedding_model"].encode.call_args[0][0]
        assert encode_call_texts == [r["text"] for r in raw]

    def test_metadata_preserved(self, patch_globals):
        raw = _make_raw_docs(2)
        patch_globals["generate_all_documents"].return_value = raw
        build_index.build_index(MagicMock())
        added_metadatas = patch_globals["collection"].add.call_args_list[0][1]["metadatas"]
        assert added_metadatas == [r["metadata"] for r in raw]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_encode_called_with_show_progress_bar(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(2)
        build_index.build_index(MagicMock())
        _, kwargs = patch_globals["embedding_model"].encode.call_args
        assert kwargs.get("show_progress_bar") is True

    def test_embeddings_passed_to_collection_add(self, patch_globals):
        n = 3
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(n)
        build_index.build_index(MagicMock())
        added = patch_globals["collection"].add.call_args_list[0][1]["embeddings"]
        # _fake_encode returns floats; tolist() converts ndarray rows to lists
        assert len(added) == n
        assert isinstance(added[0], list)


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------

class TestCollectionLifecycle:
    def test_old_collection_deleted_before_create(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(1)
        build_index.build_index(MagicMock())
        patch_globals["client"].delete_collection.assert_called_once_with(
            build_index.COLLECTION_NAME
        )

    def test_delete_failure_does_not_raise(self, patch_globals):
        patch_globals["client"].delete_collection.side_effect = Exception("not found")
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(1)
        # Should not raise
        build_index.build_index(MagicMock())

    def test_new_collection_created_with_correct_name(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(1)
        build_index.build_index(MagicMock())
        patch_globals["client"].create_collection.assert_called_once_with(
            name=build_index.COLLECTION_NAME
        )

    def test_delete_called_before_create(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(1)
        call_order = []
        patch_globals["client"].delete_collection.side_effect = lambda *a, **kw: call_order.append("delete")
        patch_globals["client"].create_collection.side_effect = lambda *a, **kw: (call_order.append("create"), patch_globals["collection"])[1]
        build_index.build_index(MagicMock())
        assert call_order == ["delete", "create"]

    def test_returns_collection(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(2)
        result = build_index.build_index(MagicMock())
        assert result is patch_globals["collection"]


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------

class TestDocumentIds:
    def test_ids_are_sequential_doc_prefix(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(4)
        build_index.build_index(MagicMock())
        # Collect all ids across batch add calls
        all_ids = []
        for c in patch_globals["collection"].add.call_args_list:
            all_ids.extend(c[1]["ids"])
        assert all_ids == ["doc-0", "doc-1", "doc-2", "doc-3"]

    def test_ids_length_matches_doc_count(self, patch_globals):
        n = 7
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(n)
        build_index.build_index(MagicMock())
        all_ids = []
        for c in patch_globals["collection"].add.call_args_list:
            all_ids.extend(c[1]["ids"])
        assert len(all_ids) == n


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:
    def test_single_batch_when_docs_under_batch_size(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(10)
        build_index.build_index(MagicMock())
        assert patch_globals["collection"].add.call_count == 1

    def test_exactly_batch_size_is_single_call(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(5000)
        build_index.build_index(MagicMock())
        assert patch_globals["collection"].add.call_count == 1

    def test_two_batches_when_docs_exceed_batch_size(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(5001)
        build_index.build_index(MagicMock())
        assert patch_globals["collection"].add.call_count == 2

    def test_three_batches_for_double_batch_size_plus_one(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(10001)
        build_index.build_index(MagicMock())
        assert patch_globals["collection"].add.call_count == 3

    def test_all_docs_added_across_batches(self, patch_globals):
        n = 5003
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(n)
        build_index.build_index(MagicMock())
        total_added = sum(
            len(c[1]["ids"]) for c in patch_globals["collection"].add.call_args_list
        )
        assert total_added == n

    def test_first_batch_has_max_batch_size_docs(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(5003)
        build_index.build_index(MagicMock())
        first_batch_ids = patch_globals["collection"].add.call_args_list[0][1]["ids"]
        assert len(first_batch_ids) == 5000

    def test_last_batch_has_remainder_docs(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(5003)
        build_index.build_index(MagicMock())
        last_batch_ids = patch_globals["collection"].add.call_args_list[-1][1]["ids"]
        assert len(last_batch_ids) == 3

    def test_empty_dataframe_no_add_called(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = []
        build_index.build_index(MagicMock())
        patch_globals["collection"].add.assert_not_called()

    def test_each_batch_add_has_ids_documents_metadatas_embeddings(self, patch_globals):
        patch_globals["generate_all_documents"].return_value = _make_raw_docs(5001)
        build_index.build_index(MagicMock())
        for add_call in patch_globals["collection"].add.call_args_list:
            kwargs = add_call[1]
            assert "ids" in kwargs
            assert "documents" in kwargs
            assert "metadatas" in kwargs
            assert "embeddings" in kwargs
