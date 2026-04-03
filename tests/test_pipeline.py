"""Unit tests for src/pipeline.py."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline


class TestBuildPrompt:
    def test_build_prompt_includes_query(self):
        query = "What is the best category by sales?"
        docs = [Document(page_content="Technology has highest sales.", metadata={})]

        prompt = pipeline.build_prompt(query, docs)

        assert query in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    def test_build_prompt_includes_all_document_contents(self):
        docs = [
            Document(page_content="Doc one content.", metadata={}),
            Document(page_content="Doc two content.", metadata={}),
        ]

        prompt = pipeline.build_prompt("q", docs)

        assert "Doc one content." in prompt
        assert "Doc two content." in prompt
        assert "Context:" in prompt

    def test_build_prompt_handles_empty_docs(self):
        prompt = pipeline.build_prompt("q", [])

        assert "Context:" in prompt
        assert "Question:" in prompt


class TestGenerateAnswer:
    def test_generate_answer_calls_llm_with_prompt(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer text"
        monkeypatch.setattr(pipeline, "llm", mock_llm)

        docs = [Document(page_content="Context one", metadata={})]
        result = pipeline.generate_answer("question", docs)

        assert result == "Answer text"
        mock_llm.invoke.assert_called_once()
        sent_prompt = mock_llm.invoke.call_args[0][0]
        assert "question" in sent_prompt
        assert "Context one" in sent_prompt

    def test_generate_answer_strips_whitespace(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  final answer\n"
        monkeypatch.setattr(pipeline, "llm", mock_llm)

        docs = [Document(page_content="ctx", metadata={})]
        result = pipeline.generate_answer("q", docs)

        assert result == "final answer"


class TestAskQuestion:
    def test_ask_question_calls_retrieve_and_generate(self, monkeypatch):
        mock_docs = [Document(page_content="retrieved", metadata={"type": "row"})]

        mock_retrieve = MagicMock(return_value=mock_docs)
        mock_generate = MagicMock(return_value="generated answer")

        monkeypatch.setattr(pipeline, "retrieve", mock_retrieve)
        monkeypatch.setattr(pipeline, "generate_answer", mock_generate)

        query = "What happened in West region?"
        vectordb = object()

        answer, docs = pipeline.ask_question(query, vectordb)

        mock_retrieve.assert_called_once_with(query, vectordb)
        mock_generate.assert_called_once_with(query, mock_docs)
        assert answer == "generated answer"
        assert docs == mock_docs

    def test_ask_question_preserves_document_list_from_retrieve(self, monkeypatch):
        docs = [
            Document(page_content="a", metadata={}),
            Document(page_content="b", metadata={}),
        ]
        monkeypatch.setattr(pipeline, "retrieve", MagicMock(return_value=docs))
        monkeypatch.setattr(pipeline, "generate_answer", MagicMock(return_value="ok"))

        answer, returned_docs = pipeline.ask_question("q", object())

        assert answer == "ok"
        assert returned_docs is docs
