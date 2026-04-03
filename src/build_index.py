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
    """
    Build a ChromaDB vector index from a sales DataFrame.

    Converts the DataFrame into text documents via generate_all_documents,
    computes sentence-transformer embeddings for every document, then stores
    them in a persistent ChromaDB collection called COLLECTION_NAME.

    The collection is always recreated from scratch so that repeated calls
    produce a clean, duplicate-free index.  Documents are written in batches
    of 5,000 to stay within ChromaDB's maximum add-batch size.

    Args:
        df (pandas.DataFrame): The cleaned sales DataFrame.  Must contain the
            columns expected by generate_all_documents (e.g. 'Order Date',
            'Sales', 'Profit', 'Region', 'Category', etc.).

    Returns:
        chromadb.Collection: The newly created and populated ChromaDB
            collection, ready for similarity queries.
    """
    raw_docs = generate_all_documents(df)

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
