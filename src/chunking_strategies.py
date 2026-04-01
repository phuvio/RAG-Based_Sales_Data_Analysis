from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def fixed_size_chunking(documents, chunk_size=1000, overlap=100):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    
    return splitter.split_documents(documents)

def recursive_chunking(documents, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)

def print_chunk_stats(chunks):
    print(f"Total chunks: {len(chunks)}")
    lengths = [len(c.page_content) for c in chunks]
    print(f"Avg length: {sum(lengths)//len(lengths)}")
    print(f"Max length: {max(lengths)}")

def smart_chunking(documents):
    chunked_docs = []
    
    for doc in documents:
        if doc.metadata["type"] == "row":
            chunked_docs.append(doc)
        else:
            chunks = recursive_chunking([doc])
            chunked_docs.extend(chunks)
    
    return chunked_docs
