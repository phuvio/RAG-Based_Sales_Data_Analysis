import streamlit as st
from langchain_core.documents import Document

from src.pipeline import ask_question
from src.build_index import COLLECTION_NAME, client, embedding_model


class ChromaRetrieverAdapter:
    """Small adapter that exposes a LangChain-like similarity_search API."""

    def __init__(self):
        self._collection = client.get_collection(name=COLLECTION_NAME)

    def similarity_search(self, query, k=20, filter=None):
        query_embedding = embedding_model.encode([query])[0].tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
        )

        docs = []
        for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
            docs.append(Document(page_content=content, metadata=metadata or {}))
        return docs


def render_app(vectordb, ask_question_fn=ask_question):
    """Render the Streamlit UI and execute the query flow when user asks one."""
    st.title("Sales Data RAG")
    query = st.text_input("Ask a question")

    if not query:
        return

    answer, docs = ask_question_fn(query, vectordb)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for doc in docs:
        st.write(doc.page_content)


def main():
    """Entry point for running the Streamlit app."""
    vectordb = ChromaRetrieverAdapter()
    render_app(vectordb)


if __name__ == "__main__":
    main()
