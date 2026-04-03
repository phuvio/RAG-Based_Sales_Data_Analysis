from langchain_ollama import OllamaLLM
from retrieval import retrieve_relevant_chunks as retrieve


llm = OllamaLLM(model="mistral-7b-instruct-v0.1.Q4_0.gguf")


def generate_answer(query, retrieved_docs):
    """
    Generate a natural language answer to the query based on the retrieved
    documents.

    Args:
        query (str): The user's natural language question.
        retrieved_docs (list[Document]): A list of relevant Document objects
            retrieved from the vector store.

    Returns:
        str: The generated answer text.
    """
    # Create a prompt that includes the query and the retrieved documents.
    prompt = build_prompt(query, retrieved_docs)
    
    # Generate and return the answer from the LLM.
    answer = llm.invoke(prompt)
    return answer.strip()

def build_prompt(query, retrieved_docs):
    """
    Build a prompt for the LLM that includes the user's query and the retrieved
    documents.

    Args:
        query (str): The user's natural language question.
        retrieved_docs (list[Document]): A list of relevant Document objects.

    Returns:
        str: The constructed prompt text.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
        SYSTEM:
        You are a retail sales analyst for Superstore (2014-2017).
        Your task is to answer questions strictly based on provided data.

        CONTEXT:
        Use ONLY the following data:
        {context}

        RULES:
        - Do NOT use knowledge outside the provided context.
        - If the answer cannot be found, respond: "Insufficient data."
        - Always support your answer with numbers from the data.
        - Combine information from multiple documents if needed.
        - Keep the answer concise and analytical.

        QUESTION:
        {query}

        ANSWER:
        """
    return prompt

def ask_question(query, vectordb):
    
    # 1. Retrieve
    docs = retrieve(query, vectordb)
    
    # 2. Generate
    answer = generate_answer(query, docs)
    
    return answer, docs
