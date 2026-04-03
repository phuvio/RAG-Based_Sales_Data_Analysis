from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import Counter
from langchain_core.documents import Document


def _to_document(item):
    """
    Coerce an arbitrary input into a LangChain Document.

    Accepted input types:
    - Document  — returned as-is (no copy).
    - dict      — expects a 'text' or 'page_content' key for the body and an
                  optional 'metadata' key; defaults to empty metadata.
    - anything else — converted to a string via str() with empty metadata.

    Args:
        item: The value to convert.

    Returns:
        Document: A LangChain Document instance.
    """
    if isinstance(item, Document):
        return item
    if isinstance(item, dict):
        text = item.get("text", item.get("page_content", ""))
        metadata = item.get("metadata", {})
        return Document(page_content=str(text), metadata=metadata)
    return Document(page_content=str(item), metadata={})


def _normalize_documents(documents):
    """
    Normalise a mixed list of inputs into a list of LangChain Documents.

    Each element is passed through _to_document, so Documents, dicts, and
    arbitrary objects are all accepted.

    Args:
        documents (list): Input items to normalise.

    Returns:
        list[Document]: A list of LangChain Document objects.
    """
    return [_to_document(doc) for doc in documents]

def fixed_size_chunking(documents, chunk_size=1000, overlap=100):
    """
    Split documents into fixed-size character windows.

    Uses a single empty-string separator so chunks are cut at raw character
    boundaries regardless of whitespace or punctuation.  This guarantees that
    chunk sizes change consistently even for text without newlines.

    Args:
        documents (list): Input documents (Documents, dicts, or other objects).
        chunk_size (int): Maximum number of characters per chunk. Default 1000.
        overlap (int): Number of characters to overlap between adjacent chunks.
            Default 100.

    Returns:
        list[Document]: The split Document chunks with metadata preserved.
    """
    # Use raw character windows so chunking changes even when text has no newlines.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[""],
    )

    return splitter.split_documents(_normalize_documents(documents))

def recursive_chunking(documents, chunk_size=1000, overlap=100):
    """
    Split documents using a hierarchy of natural-language separators.

    Tries to break on paragraph boundaries first ("\\n\\n"), then line breaks,
    sentence endings, spaces, and finally raw characters.  This produces
    chunks that align better with sentence and paragraph structure than
    fixed_size_chunking.

    Args:
        documents (list): Input documents (Documents, dicts, or other objects).
        chunk_size (int): Maximum number of characters per chunk. Default 1000.
        overlap (int): Number of characters to overlap between adjacent chunks.
            Default 100.

    Returns:
        list[Document]: The split Document chunks with metadata preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(_normalize_documents(documents))

def smart_chunking(documents):
    """
    Apply document-type-aware chunking.

    Documents whose metadata 'type' is 'row' represent individual transaction
    records and are short by design, so they are kept intact.  All other
    document types (summaries, trends, etc.) are passed through
    recursive_chunking to split long texts into retrieval-friendly pieces.

    Args:
        documents (list): Input documents (Documents, dicts, or other objects).

    Returns:
        list[Document]: Chunked documents; 'row' docs are unchanged, others
            are split by recursive_chunking.
    """
    chunked_docs = []

    for doc in _normalize_documents(documents):
        if doc.metadata.get("type") == "row":
            chunked_docs.append(doc)
        else:
            chunks = recursive_chunking([doc])
            chunked_docs.extend(chunks)
    
    return chunked_docs

def print_stats(name, chunks):
    """
    Print basic length statistics for a list of chunks.

    Outputs total count, average, minimum, and maximum character lengths to
    stdout.  Useful for quick inspection during experimentation.

    Args:
        name (str): A label identifying the chunking strategy (not printed
            in the current implementation but kept for caller context).
        chunks (list[Document]): The chunks to analyse.
    """
    lengths = [len(c.page_content) for c in chunks]

    print(f"Total chunks: {len(chunks)}")
    print(f"Avg length: {sum(lengths)//len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")

def print_type_distribution(chunks):
    """
    Print a frequency count of document types present in a chunk list.

    Reads the 'type' field from each chunk's metadata (defaulting to
    'unknown') and prints the Counter to stdout.

    Args:
        chunks (list[Document]): The chunks to inspect.
    """
    types = [c.metadata.get("type", "unknown") for c in chunks]
    print("Type distribution:", Counter(types))

def inspect_chunks(chunks, n=2):
    """
    Print the first n chunks for visual inspection.

    Each chunk is displayed with a numbered header, the first 300 characters
    of its content, and its full metadata dict.

    Args:
        chunks (list[Document]): The chunks to display.
        n (int): Number of chunks to print. Default 2.
    """
    for i in range(n):
        print(f"\n--- Chunk {i} ---")
        print(chunks[i].page_content[:300])
        print("Metadata:", chunks[i].metadata)

def check_sentence_breaks(chunks):
    """
    Count and print how many chunks end mid-sentence.

    A chunk is considered 'clean' if its stripped text ends with '.', '!', or
    '?'.  Prints a summary line in the form
    "Chunks ending mid-sentence: <bad>/<total>" to stdout.

    Args:
        chunks (list[Document]): The chunks to check.
    """
    bad = 0
    for c in chunks:
        text = c.page_content.strip()
        if not text.endswith((".", "!", "?")):
            bad += 1
    print(f"Chunks ending mid-sentence: {bad}/{len(chunks)}")

def chunks_per_doc(chunks):
    """
    Print the total number of chunks as a rough proxy for chunks-per-document.

    Note: this is an approximation that counts total chunks rather than
    computing a per-source-document breakdown.

    Args:
        chunks (list[Document]): The chunks to count.
    """
    ids = [id(c.metadata) for c in chunks]
    print("Approx chunks per original doc:", len(chunks))
