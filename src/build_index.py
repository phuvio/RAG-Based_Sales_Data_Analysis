from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import chromadb
from generate_documents import generate_all_documents
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "chroma_db"
COLLECTION_NAME = "superstore_index"

client = chromadb.PersistentClient(path=str(DB_PATH))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(df):

    # Generoi dokumentit
    raw_docs = generate_all_documents(df)

    # Muunna LangChain Document -objekteiksi
    documents = [
        Document(
            page_content=doc["text"],
            metadata=doc["metadata"]
        )
        for doc in raw_docs
    ]

    print(f"Total documents: {len(documents)}")

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"doc-{i}" for i in range(len(documents))]

    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

    # Recreate collection to avoid duplicate docs on repeated runs.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # Chroma enforces a maximum add batch size.
    batch_size = 5000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print(f"Index built and saved to {DB_PATH}")
    return collection


if __name__ == "__main__":
    import pandas as pd

    # Resolve data path relative to this file so script works from any cwd.
    data_path = Path(__file__).resolve().parents[1] / "data" / "cleaned_superstore.csv"
    df = pd.read_csv(data_path)
    collection = build_index(df)

    query_embedding = embedding_model.encode(["best category by sales"])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        print(doc)
        print(metadata)
