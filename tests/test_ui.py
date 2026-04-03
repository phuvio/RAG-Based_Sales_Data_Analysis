"""Unit tests for src/ui.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import src.ui as ui


class TestRenderApp:
    def test_renders_title_and_input(self, monkeypatch):
        st_mock = MagicMock()
        st_mock.text_input.return_value = ""
        monkeypatch.setattr(ui, "st", st_mock)

        ui.render_app(vectordb=object(), ask_question_fn=MagicMock())

        st_mock.title.assert_called_once_with("Sales Data RAG")
        st_mock.text_input.assert_called_once_with("Ask a question")

    def test_does_not_call_pipeline_when_query_empty(self, monkeypatch):
        st_mock = MagicMock()
        st_mock.text_input.return_value = ""
        ask_mock = MagicMock()

        monkeypatch.setattr(ui, "st", st_mock)

        ui.render_app(vectordb=object(), ask_question_fn=ask_mock)

        ask_mock.assert_not_called()
        st_mock.subheader.assert_not_called()

    def test_calls_pipeline_and_renders_answer_and_sources(self, monkeypatch):
        st_mock = MagicMock()
        st_mock.text_input.return_value = "best category"
        ask_mock = MagicMock(
            return_value=(
                "Technology is the top category.",
                [
                    Document(page_content="Source one", metadata={}),
                    Document(page_content="Source two", metadata={}),
                ],
            )
        )

        monkeypatch.setattr(ui, "st", st_mock)

        vectordb = object()
        ui.render_app(vectordb=vectordb, ask_question_fn=ask_mock)

        ask_mock.assert_called_once_with("best category", vectordb)
        st_mock.subheader.assert_any_call("Answer")
        st_mock.subheader.assert_any_call("Sources")
        st_mock.write.assert_any_call("Technology is the top category.")
        st_mock.write.assert_any_call("Source one")
        st_mock.write.assert_any_call("Source two")


class TestChromaRetrieverAdapter:
    def test_similarity_search_queries_collection_and_maps_documents(self, monkeypatch):
        fake_collection = MagicMock()
        fake_collection.query.return_value = {
            "documents": [["Doc A", "Doc B"]],
            "metadatas": [[{"type": "row"}, {"type": "category_summary"}]],
        }

        fake_client = MagicMock()
        fake_client.get_collection.return_value = fake_collection

        fake_embedding_model = MagicMock()
        fake_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        monkeypatch.setattr(ui, "client", fake_client)
        monkeypatch.setattr(ui, "embedding_model", fake_embedding_model)

        adapter = ui.ChromaRetrieverAdapter()
        docs = adapter.similarity_search("query text", k=5, filter={"type": {"$in": ["row"]}})

        fake_client.get_collection.assert_called_once_with(name=ui.COLLECTION_NAME)
        fake_embedding_model.encode.assert_called_once_with(["query text"])
        fake_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5,
            where={"type": {"$in": ["row"]}},
        )

        assert len(docs) == 2
        assert docs[0].page_content == "Doc A"
        assert docs[0].metadata["type"] == "row"
        assert docs[1].page_content == "Doc B"
        assert docs[1].metadata["type"] == "category_summary"

    def test_similarity_search_handles_missing_metadata(self, monkeypatch):
        fake_collection = MagicMock()
        fake_collection.query.return_value = {
            "documents": [["Doc A"]],
            "metadatas": [[None]],
        }

        fake_client = MagicMock()
        fake_client.get_collection.return_value = fake_collection

        fake_embedding_model = MagicMock()
        fake_embedding_model.encode.return_value = np.array([[0.9, 0.8]])

        monkeypatch.setattr(ui, "client", fake_client)
        monkeypatch.setattr(ui, "embedding_model", fake_embedding_model)

        adapter = ui.ChromaRetrieverAdapter()
        docs = adapter.similarity_search("query")

        assert len(docs) == 1
        assert docs[0].metadata == {}


class TestMain:
    def test_main_constructs_adapter_and_renders_app(self, monkeypatch):
        fake_adapter_instance = object()
        adapter_cls = MagicMock(return_value=fake_adapter_instance)
        render_mock = MagicMock()

        monkeypatch.setattr(ui, "ChromaRetrieverAdapter", adapter_cls)
        monkeypatch.setattr(ui, "render_app", render_mock)

        ui.main()

        adapter_cls.assert_called_once_with()
        render_mock.assert_called_once_with(fake_adapter_instance)
