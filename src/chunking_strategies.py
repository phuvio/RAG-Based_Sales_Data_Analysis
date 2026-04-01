from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
from langchain_core.documents import Document


def _to_document(item):
    if isinstance(item, Document):
        return item
    if isinstance(item, dict):
        text = item.get("text", item.get("page_content", ""))
        metadata = item.get("metadata", {})
        return Document(page_content=str(text), metadata=metadata)
    return Document(page_content=str(item), metadata={})


def _normalize_documents(documents):
    return [_to_document(doc) for doc in documents]

def fixed_size_chunking(documents, chunk_size=1000, overlap=100):
    # Use raw character windows so chunking changes even when text has no newlines.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[""],
    )

    return splitter.split_documents(_normalize_documents(documents))

def recursive_chunking(documents, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(_normalize_documents(documents))

def smart_chunking(documents):
    chunked_docs = []

    for doc in _normalize_documents(documents):
        if doc.metadata.get("type") == "row":
            chunked_docs.append(doc)
        else:
            chunks = recursive_chunking([doc])
            chunked_docs.extend(chunks)
    
    return chunked_docs

def print_stats(name, chunks):
    lengths = [len(c.page_content) for c in chunks]

    print(f"Total chunks: {len(chunks)}")
    print(f"Avg length: {sum(lengths)//len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")

def print_type_distribution(chunks):
    types = [c.metadata.get("type", "unknown") for c in chunks]
    print("Type distribution:", Counter(types))

def inspect_chunks(chunks, n=2):
    for i in range(n):
        print(f"\n--- Chunk {i} ---")
        print(chunks[i].page_content[:300])
        print("Metadata:", chunks[i].metadata)

def check_sentence_breaks(chunks):
    bad = 0
    for c in chunks:
        text = c.page_content.strip()
        if not text.endswith((".", "!", "?")):
            bad += 1
    print(f"Chunks ending mid-sentence: {bad}/{len(chunks)}")

def chunks_per_doc(chunks):
    ids = [id(c.metadata) for c in chunks]
    print("Approx chunks per original doc:", len(chunks))
