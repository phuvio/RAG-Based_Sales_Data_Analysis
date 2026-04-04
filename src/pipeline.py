from langchain_ollama import OllamaLLM
from retrieval import retrieve_relevant_chunks as retrieve


llm = OllamaLLM(model="mistral")
MAX_CHAT_HISTORY_PROMPTS = 3
chat_history = []


def generate_answer(query, retrieved_docs, chat_history):
    """
    Generate a natural language answer to the query based on the retrieved
    documents and chat history.

    Args:
        query (str): The user's natural language question.
        retrieved_docs (list[Document]): A list of relevant Document objects
            retrieved from the vector store.
        chat_history (list[str]): A list of previous chat messages for context.

    Returns:
        str: The generated answer text.
    """
    # Create a prompt that includes the query and the retrieved documents.
    prompt = build_prompt(query, retrieved_docs, chat_history)
    
    # Generate and return the answer from the LLM.
    try:
        answer = llm.invoke(prompt)
        return answer.strip()
    except Exception as exc:
        # Most common runtime failure is Ollama not running on localhost:11434.
        return (
            "LLM connection failed. Start Ollama and ensure the selected model is available. "
            f"Details: {exc}"
        )

def build_prompt(query, retrieved_docs, chat_history):
    """
    Build a prompt for the LLM that includes the user's query and the retrieved
    documents.

    Args:
        query (str): The user's natural language question.
        retrieved_docs (list[Document]): A list of relevant Document objects.
        chat_history (list[str]): A list of previous chat messages.

    Returns:
        str: The constructed prompt text.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    history = "\n\n".join(chat_history)

    prompt = f"""
        SYSTEM:
        You are a retail sales analyst for Superstore (2014-2017).
        Your task is to answer questions strictly based on provided data.

        CONTEXT:
        Use ONLY the following data:
        {context}
        
        CHAT HISTORY:
        {history}

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
    """
    Process a user query by retrieving relevant documents and generating an answer.
    Maintains chat history for multi-turn conversations, keeping only the
    last 3 prompt-answer pairs.

    Args:
        query (str): The user's natural language question.
        vectordb: The vector database to retrieve documents from.

    Returns:
        tuple: (answer, docs) where answer is the generated response text and
               docs is the list of retrieved Document objects.
    """
    # 1. Retrieve
    docs = retrieve(query, vectordb)
    
    # 2. Generate
    answer = generate_answer(query, docs, chat_history)

    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")
    max_history_entries = MAX_CHAT_HISTORY_PROMPTS * 2
    if len(chat_history) > max_history_entries:
        del chat_history[:-max_history_entries]
    
    return answer, docs
