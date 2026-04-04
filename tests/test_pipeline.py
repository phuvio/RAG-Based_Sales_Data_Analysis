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
        chat_history = []

        prompt = pipeline.build_prompt(query, docs, chat_history)

        assert query in prompt
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt
        assert "SYSTEM:" in prompt

    def test_build_prompt_includes_all_document_contents(self):
        docs = [
            Document(page_content="Doc one content.", metadata={}),
            Document(page_content="Doc two content.", metadata={}),
        ]
        chat_history = []

        prompt = pipeline.build_prompt("q", docs, chat_history)

        assert "Doc one content." in prompt
        assert "Doc two content." in prompt
        assert "CONTEXT:" in prompt

    def test_build_prompt_handles_empty_docs(self):
        chat_history = []
        prompt = pipeline.build_prompt("q", [], chat_history)

        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt

    def test_build_prompt_includes_chat_history(self):
        docs = [Document(page_content="Sales data.", metadata={})]
        chat_history = ["User: Previous question", "Assistant: Previous answer"]

        prompt = pipeline.build_prompt("What about profits?", docs, chat_history)

        assert "CHAT HISTORY:" in prompt
        assert "Previous question" in prompt
        assert "Previous answer" in prompt

    def test_build_prompt_handles_empty_chat_history(self):
        docs = [Document(page_content="Data.", metadata={})]
        chat_history = []

        prompt = pipeline.build_prompt("question", docs, chat_history)

        assert "CHAT HISTORY:" in prompt


class TestGenerateAnswer:
    def test_generate_answer_calls_llm_with_prompt(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer text"
        monkeypatch.setattr(pipeline, "llm", mock_llm)

        docs = [Document(page_content="Context one", metadata={})]
        chat_history = []
        result = pipeline.generate_answer("question", docs, chat_history)

        assert result == "Answer text"
        mock_llm.invoke.assert_called_once()
        sent_prompt = mock_llm.invoke.call_args[0][0]
        assert "question" in sent_prompt
        assert "Context one" in sent_prompt

    def test_generate_answer_includes_chat_history(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer text"
        monkeypatch.setattr(pipeline, "llm", mock_llm)

        docs = [Document(page_content="Data", metadata={})]
        chat_history = ["User: What is Q1 sales?", "Assistant: Q1 sales were 100k"]
        result = pipeline.generate_answer("What about Q2?", docs, chat_history)

        sent_prompt = mock_llm.invoke.call_args[0][0]
        assert "Q1 sales" in sent_prompt
        assert "100k" in sent_prompt

    def test_generate_answer_strips_whitespace(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "  final answer\n"
        monkeypatch.setattr(pipeline, "llm", mock_llm)

        docs = [Document(page_content="ctx", metadata={})]
        chat_history = []
        result = pipeline.generate_answer("q", docs, chat_history)

        assert result == "final answer"

    def test_generate_answer_returns_message_when_llm_fails(self, monkeypatch):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("connection refused")
        monkeypatch.setattr(pipeline, "llm", mock_llm)

        docs = [Document(page_content="ctx", metadata={})]
        chat_history = []
        result = pipeline.generate_answer("q", docs, chat_history)

        assert "LLM connection failed" in result
        assert "connection refused" in result


class TestAskQuestion:
    def test_ask_question_calls_retrieve_and_generate(self, monkeypatch):
        mock_docs = [Document(page_content="retrieved", metadata={"type": "row"})]

        mock_retrieve = MagicMock(return_value=mock_docs)
        mock_generate = MagicMock(return_value="generated answer")

        monkeypatch.setattr(pipeline, "retrieve", mock_retrieve)
        monkeypatch.setattr(pipeline, "generate_answer", mock_generate)
        
        # Reset chat_history before test
        pipeline.chat_history.clear()

        query = "What happened in West region?"
        vectordb = object()

        answer, docs = pipeline.ask_question(query, vectordb)

        mock_retrieve.assert_called_once_with(query, vectordb)
        # Verify generate_answer was called with the query, docs, and chat_history
        call_args = mock_generate.call_args[0]
        assert call_args[0] == query  # query
        assert call_args[1] == mock_docs  # docs
        assert isinstance(call_args[2], list)  # chat_history
        
        assert answer == "generated answer"
        assert docs == mock_docs

    def test_ask_question_updates_chat_history(self, monkeypatch):
        mock_docs = [Document(page_content="retrieved", metadata={"type": "row"})]

        mock_retrieve = MagicMock(return_value=mock_docs)
        mock_generate = MagicMock(return_value="The answer")

        monkeypatch.setattr(pipeline, "retrieve", mock_retrieve)
        monkeypatch.setattr(pipeline, "generate_answer", mock_generate)
        
        # Reset chat_history
        pipeline.chat_history.clear()

        query = "What is total sales?"
        vectordb = object()

        answer, docs = pipeline.ask_question(query, vectordb)

        # Verify chat_history was updated
        assert len(pipeline.chat_history) == 2
        assert "User: What is total sales?" in pipeline.chat_history
        assert "Assistant: The answer" in pipeline.chat_history

    def test_ask_question_preserves_chat_history_across_calls(self, monkeypatch):
        mock_docs = [Document(page_content="retrieved", metadata={"type": "row"})]

        mock_retrieve = MagicMock(return_value=mock_docs)
        mock_generate = MagicMock(side_effect=["First answer", "Second answer"])

        monkeypatch.setattr(pipeline, "retrieve", mock_retrieve)
        monkeypatch.setattr(pipeline, "generate_answer", mock_generate)
        
        # Reset chat_history
        pipeline.chat_history.clear()

        # First question
        answer1, docs1 = pipeline.ask_question("Question 1?", object())
        assert answer1 == "First answer"
        assert len(pipeline.chat_history) == 2

        # Second question
        answer2, docs2 = pipeline.ask_question("Question 2?", object())
        assert answer2 == "Second answer"
        assert len(pipeline.chat_history) == 4
        assert "User: Question 1?" in pipeline.chat_history
        assert "Assistant: First answer" in pipeline.chat_history

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
