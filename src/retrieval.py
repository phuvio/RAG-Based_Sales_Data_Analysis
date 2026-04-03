from langchain_core.documents import Document


def detect_intent(query: str) -> list[str] | None:
    """
    Detects the intent of the query using keyword matching and returns a list
    of document types (matching metadata 'type' values from generate_documents.py)
    that are most likely to contain a relevant answer.

    Rules are evaluated in priority order — most specific first.
    Returns None when no intent can be determined (triggers full vector search).

    Args:
        query: The user's natural language question.

    Returns:
        A list of document type strings, or None for an open search.
    """
    q = query.lower()

    # ------------------------------------------------------------------ #
    # 1. Comparisons (most specific — check before generic category/region)
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("technology vs", "vs furniture", "technology and furniture",
                             "compare categor", "category vs category")):
        return ["category_comparison", "category_summary"]

    if any(p in q for p in ("west vs east", "east vs west", "west and east",
                             "compare region", "region vs region")):
        return ["region_comparison", "region_summary"]

    # ------------------------------------------------------------------ #
    # 2. Specific "top / best / highest" superlatives
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("best month", "top month", "highest month",
                             "peak month", "busiest month", "worst month")):
        return ["top_months", "monthly_summary"]

    if any(p in q for p in ("best sub", "top sub", "most profitable sub",
                             "highest sub", "top-sub", "top sub-cat")):
        return ["top_subcategory", "subcategory_summary"]

    if any(p in q for p in ("best category", "top category", "highest category",
                             "most revenue category", "best selling category",
                             "most popular category")):
        return ["top_category", "category_summary"]

    if any(p in q for p in ("best region", "top region", "highest region",
                             "most revenue region", "leading region")):
        return ["top_region", "region_summary"]

    if any(p in q for p in ("best customer", "top customer", "biggest customer",
                             "most valuable customer", "highest spending customer")):
        return ["top_customers"]

    if any(p in q for p in ("best segment", "top segment", "leading segment",
                             "worst segment", "lowest segment", "compare segment")):
        return ["segment_comparison", "segment_summary"]

    # ------------------------------------------------------------------ #
    # 3. Discount queries
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("discount", "markdown", "price reduction",
                             "promotional", "on sale", "reduced price")):
        return ["discount_category", "category_summary", "subcategory_summary"]

    # ------------------------------------------------------------------ #
    # 4. Category × time (trend per category)
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("category trend", "category over time",
                             "category growth", "category by year",
                             "how has category", "category performance")):
        return ["category_trend", "yearly_summary"]

    # ------------------------------------------------------------------ #
    # 5. Region × category cross queries
    # ------------------------------------------------------------------ #
    if ("region" in q and "category" in q) or ("category" in q and "region" in q):
        return ["region_category_summary", "category_summary", "region_summary"]

    # ------------------------------------------------------------------ #
    # 6. Time / trend queries
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("trend", "over time", "growth", "decline",
                             "year on year", "yoy", "annual", "yearly",
                             "how have sales", "how has profit")):
        return ["trend_summary", "yearly_summary", "monthly_summary"]

    if any(p in q for p in ("monthly", "per month", "month by month",
                             "each month", "which month")):
        return ["monthly_summary", "top_months"]

    if any(p in q for p in ("yearly", "per year", "year by year",
                             "each year", "which year", "annual")):
        return ["yearly_summary", "trend_summary"]

    # ------------------------------------------------------------------ #
    # 7. Geographic granularity
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("city", "cities", "town", "metropolitan")):
        return ["city_summary"]

    if any(p in q for p in ("state", "province", "territory")):
        return ["state_summary"]

    if any(p in q for p in ("region", "area", "geographic", "geography",
                             "east", "west", "south", "central")):
        return ["region_summary", "top_region"]

    # ------------------------------------------------------------------ #
    # 8. Product hierarchy
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("sub-category", "subcategory", "sub category",
                             "furniture sub", "technology sub", "office sub")):
        return ["subcategory_summary", "top_subcategory"]

    if any(p in q for p in ("category", "furniture", "technology",
                             "office supplies")):
        return ["category_summary", "top_category"]

    # ------------------------------------------------------------------ #
    # 9. Customer segment
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("segment", "consumer", "corporate",
                             "home office", "b2b", "b2c")):
        return ["segment_summary", "segment_comparison"]

    # ------------------------------------------------------------------ #
    # 10. Profit / margin
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("profit", "profitable", "margin", "earnings",
                             "net", "loss", "losing")):
        return ["top_subcategory", "subcategory_summary",
                "category_summary", "region_summary"]

    # ------------------------------------------------------------------ #
    # 11. Customer queries
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("customer", "buyer", "client", "who bought",
                             "who purchased", "purchaser")):
        return ["top_customers"]

    # ------------------------------------------------------------------ #
    # 12. Order / product / transaction level
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("order", "purchase", "transaction", "product",
                             "item", "sku", "quantity", "units sold")):
        return ["row"]

    # ------------------------------------------------------------------ #
    # 13. Sales (generic fallback — broad search)
    # ------------------------------------------------------------------ #
    if any(p in q for p in ("sales", "revenue", "income")):
        return ["yearly_summary", "category_summary", "region_summary"]

    return None

def filter_by_intent(chunks: list[Document], intent_types: list[str]) -> list[Document]:
    """
    Filters a list of Document chunks to only those whose metadata 'type' matches
    one of the intent_types. If intent_types is None, returns the original list.

    Args:
        chunks: A list of Document objects with metadata.
        intent_types: A list of strings representing the 'type' values to filter by.

    Returns:
        A filtered list of Document objects matching the intent types.
    """
    if intent_types is None:
        return chunks

    filtered = [c for c in chunks if c.metadata.get("type") in intent_types]
    print(f"Filtered {len(chunks)} chunks to {len(filtered)} based on intent types: {intent_types}")
    return filtered

def retrieve_relevant_chunks(query: str, vectordb) -> list[Document]:
    """
    Main retrieval function that takes a user query and a ChromaDB collection,
    detects the intent of the query, and returns a filtered list of relevant chunks.

    Args:
        query: The user's natural language question.
        vectordb: The vector database instance to query (LangChain Chroma wrapper).

    Returns:
        A list of Document objects relevant to the query.
    """
    intent_types = detect_intent(query)

    if intent_types:
        # ChromaDB where-filter: match any of the detected type values.
        where_filter = {"type": {"$in": intent_types}}
        results = vectordb.similarity_search(
            query,
            k=5,
            filter=where_filter,
        )

        if len(results) < 3:
            results = vectordb.similarity_search(query, k=5)
    else:
        results = vectordb.similarity_search(query, k=5)

    return results
