"""
Unit tests for src/retrieval.py

Covers:
- detect_intent: every intent group, case-insensitivity, priority ordering,
  None fallback, and return type.
- filter_by_intent: filtering, None pass-through, empty list/types, partial
  overlap.
- retrieve_relevant_chunks: end-to-end integration of both sub-functions.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make src/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from langchain_core.documents import Document
from retrieval import detect_intent, filter_by_intent, retrieve_relevant_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(doc_type: str, content: str = "test") -> Document:
    return Document(page_content=content, metadata={"type": doc_type})


# ---------------------------------------------------------------------------
# detect_intent — return type contract
# ---------------------------------------------------------------------------

class TestDetectIntentReturnType:
    def test_returns_list_or_none(self):
        result = detect_intent("trend over time")
        assert isinstance(result, list)

    def test_returns_none_for_unknown_query(self):
        assert detect_intent("hello world") is None

    def test_returns_none_for_empty_string(self):
        assert detect_intent("") is None

    def test_list_is_non_empty_when_matched(self):
        result = detect_intent("what is the trend")
        assert result and len(result) > 0


# ---------------------------------------------------------------------------
# detect_intent — case insensitivity
# ---------------------------------------------------------------------------

class TestDetectIntentCaseInsensitivity:
    def test_uppercase_trend(self):
        assert detect_intent("TREND in sales") == detect_intent("trend in sales")

    def test_mixed_case_category(self):
        assert detect_intent("Best Category") == detect_intent("best category")

    def test_mixed_case_discount(self):
        assert detect_intent("What is the DISCOUNT rate") == detect_intent("what is the discount rate")


# ---------------------------------------------------------------------------
# detect_intent — Group 1: comparisons (highest priority)
# ---------------------------------------------------------------------------

class TestDetectIntentComparisons:
    def test_technology_vs_furniture(self):
        result = detect_intent("technology vs furniture in sales")
        assert "category_comparison" in result

    def test_vs_furniture_variant(self):
        result = detect_intent("how does hardware do vs furniture")
        assert "category_comparison" in result

    def test_compare_categories(self):
        result = detect_intent("compare categories by revenue")
        assert "category_comparison" in result

    def test_west_vs_east(self):
        result = detect_intent("west vs east region performance")
        assert "region_comparison" in result

    def test_east_vs_west(self):
        result = detect_intent("east vs west")
        assert "region_comparison" in result

    def test_compare_region(self):
        result = detect_intent("compare region profit")
        assert "region_comparison" in result

    # Priority: comparison must beat generic "category" / "region" matches
    def test_category_comparison_takes_priority_over_category(self):
        result = detect_intent("technology vs furniture")
        assert result[0] == "category_comparison"

    def test_region_comparison_takes_priority_over_region(self):
        result = detect_intent("west vs east")
        assert result[0] == "region_comparison"


# ---------------------------------------------------------------------------
# detect_intent — Group 2: top / best superlatives
# ---------------------------------------------------------------------------

class TestDetectIntentSuperlatives:
    def test_best_month(self):
        result = detect_intent("what was the best month for sales")
        assert "top_months" in result

    def test_peak_month(self):
        result = detect_intent("peak month of the year")
        assert "top_months" in result

    def test_top_subcategory(self):
        result = detect_intent("which is the top sub-category")
        assert "top_subcategory" in result

    def test_most_profitable_sub(self):
        result = detect_intent("most profitable sub category")
        assert "top_subcategory" in result

    def test_best_category(self):
        result = detect_intent("best category by sales")
        assert "top_category" in result

    def test_top_category(self):
        result = detect_intent("which is the top category")
        assert "top_category" in result

    def test_best_region(self):
        result = detect_intent("what is the best region")
        assert "top_region" in result

    def test_leading_region(self):
        result = detect_intent("leading region for revenue")
        assert "top_region" in result

    def test_top_customer(self):
        result = detect_intent("show me the top customer")
        assert "top_customers" in result

    def test_most_valuable_customer(self):
        result = detect_intent("most valuable customer by spend")
        assert "top_customers" in result

    def test_best_segment(self):
        result = detect_intent("which is the best segment")
        assert "segment_comparison" in result

    def test_worst_segment(self):
        result = detect_intent("worst segment performance")
        assert "segment_comparison" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 3: discounts
# ---------------------------------------------------------------------------

class TestDetectIntentDiscount:
    def test_discount_keyword(self):
        result = detect_intent("what is the average discount")
        assert "discount_category" in result

    def test_markdown(self):
        result = detect_intent("markdown applied to furniture")
        assert "discount_category" in result

    def test_on_sale(self):
        result = detect_intent("items that were on sale")
        assert "discount_category" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 4: category trend over time
# ---------------------------------------------------------------------------

class TestDetectIntentCategoryTrend:
    def test_category_trend(self):
        result = detect_intent("category trend over the years")
        assert "category_trend" in result

    def test_category_growth(self):
        result = detect_intent("category growth per year")
        assert "category_trend" in result

    def test_category_by_year(self):
        result = detect_intent("category by year breakdown")
        assert "category_trend" in result

    # Priority: category trend must beat generic time/trend matches
    def test_category_trend_beats_trend(self):
        result = detect_intent("category trend")
        assert result[0] == "category_trend"


# ---------------------------------------------------------------------------
# detect_intent — Group 5: region × category cross
# ---------------------------------------------------------------------------

class TestDetectIntentRegionCategory:
    def test_region_and_category(self):
        result = detect_intent("sales by region and category")
        assert "region_category_summary" in result

    def test_category_in_region(self):
        result = detect_intent("which category performs best in each region")
        assert "region_category_summary" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 6: time / trend
# ---------------------------------------------------------------------------

class TestDetectIntentTime:
    def test_trend(self):
        result = detect_intent("sales trend")
        assert "trend_summary" in result

    def test_over_time(self):
        result = detect_intent("how have profits changed over time")
        assert "trend_summary" in result

    def test_yoy(self):
        result = detect_intent("yoy revenue comparison")
        assert "trend_summary" in result

    def test_how_have_sales(self):
        result = detect_intent("how have sales evolved")
        assert "trend_summary" in result

    def test_monthly(self):
        result = detect_intent("monthly breakdown of sales")
        assert "monthly_summary" in result

    def test_per_month(self):
        result = detect_intent("profit per month")
        assert "monthly_summary" in result

    def test_each_month(self):
        result = detect_intent("show sales for each month")
        assert "monthly_summary" in result

    def test_yearly(self):
        result = detect_intent("yearly revenue totals")
        assert "yearly_summary" in result

    def test_per_year(self):
        result = detect_intent("profit per year")
        assert "yearly_summary" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 7: geographic
# ---------------------------------------------------------------------------

class TestDetectIntentGeographic:
    def test_city(self):
        result = detect_intent("sales in each city")
        assert result == ["city_summary"]

    def test_cities(self):
        result = detect_intent("top cities by revenue")
        assert result == ["city_summary"]

    def test_state(self):
        result = detect_intent("revenue by state")
        assert result == ["state_summary"]

    def test_province(self):
        result = detect_intent("breakdown by province")
        assert result == ["state_summary"]

    def test_region(self):
        result = detect_intent("which region had the highest sales")
        assert "region_summary" in result

    def test_geographic(self):
        result = detect_intent("show me a geographic breakdown")
        assert "region_summary" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 8: product hierarchy
# ---------------------------------------------------------------------------

class TestDetectIntentProductHierarchy:
    def test_subcategory(self):
        result = detect_intent("sales by subcategory")
        assert "subcategory_summary" in result

    def test_sub_category_hyphen(self):
        result = detect_intent("sub-category profit analysis")
        assert "subcategory_summary" in result

    def test_category_generic(self):
        result = detect_intent("category breakdown")
        assert "category_summary" in result

    def test_furniture(self):
        result = detect_intent("how is furniture performing")
        assert "category_summary" in result

    def test_technology(self):
        result = detect_intent("technology sales")
        assert "category_summary" in result

    def test_office_supplies(self):
        result = detect_intent("office supplies revenue")
        assert "category_summary" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 9: segment
# ---------------------------------------------------------------------------

class TestDetectIntentSegment:
    def test_segment(self):
        result = detect_intent("which segment buys the most")
        assert "segment_summary" in result

    def test_consumer(self):
        result = detect_intent("consumer buying patterns")
        assert "segment_summary" in result

    def test_corporate(self):
        result = detect_intent("corporate orders breakdown")
        assert "segment_summary" in result

    def test_home_office(self):
        result = detect_intent("home office segment")
        assert "segment_summary" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 10: profit / margin
# ---------------------------------------------------------------------------

class TestDetectIntentProfit:
    def test_profit(self):
        result = detect_intent("overall profit figures")
        assert "category_summary" in result

    def test_profitable(self):
        result = detect_intent("most profitable products")
        assert "top_subcategory" in result

    def test_loss(self):
        result = detect_intent("categories running at a loss")
        assert "category_summary" in result

    def test_margin(self):
        result = detect_intent("profit margin by category")
        assert "category_summary" in result


# ---------------------------------------------------------------------------
# detect_intent — Group 11: customer
# ---------------------------------------------------------------------------

class TestDetectIntentCustomer:
    def test_customer(self):
        result = detect_intent("who is our biggest customer")
        assert result == ["top_customers"]

    def test_buyer(self):
        result = detect_intent("biggest buyer last year")
        assert result == ["top_customers"]

    def test_who_purchased(self):
        result = detect_intent("who purchased the most")
        assert result == ["top_customers"]


# ---------------------------------------------------------------------------
# detect_intent — Group 12: order / row level
# ---------------------------------------------------------------------------

class TestDetectIntentRowLevel:
    def test_order(self):
        result = detect_intent("show me individual orders")
        assert result == ["row"]

    def test_product(self):
        result = detect_intent("details for a specific product")
        assert result == ["row"]

    def test_quantity(self):
        result = detect_intent("quantity of items per order")
        assert result == ["row"]

    def test_units_sold(self):
        result = detect_intent("units sold per transaction")
        assert result == ["row"]


# ---------------------------------------------------------------------------
# detect_intent — Group 13: generic sales fallback
# ---------------------------------------------------------------------------

class TestDetectIntentSalesFallback:
    def test_sales(self):
        result = detect_intent("total sales figures")
        assert "yearly_summary" in result

    def test_revenue(self):
        result = detect_intent("overall revenue")
        assert "yearly_summary" in result

    def test_income(self):
        result = detect_intent("income breakdown")
        assert "yearly_summary" in result


# ---------------------------------------------------------------------------
# filter_by_intent
# ---------------------------------------------------------------------------

class TestFilterByIntent:
    def _make_chunks(self):
        return [
            _doc("yearly_summary", "2021 sales"),
            _doc("monthly_summary", "Jan sales"),
            _doc("category_summary", "tech sales"),
            _doc("region_summary", "west sales"),
            _doc("row", "single order"),
        ]

    def test_filters_to_matching_types(self):
        chunks = self._make_chunks()
        result = filter_by_intent(chunks, ["yearly_summary", "monthly_summary"])
        types = [c.metadata["type"] for c in result]
        assert set(types) == {"yearly_summary", "monthly_summary"}

    def test_none_returns_all_chunks(self):
        chunks = self._make_chunks()
        result = filter_by_intent(chunks, None)
        assert result == chunks

    def test_single_type_match(self):
        chunks = self._make_chunks()
        result = filter_by_intent(chunks, ["row"])
        assert len(result) == 1
        assert result[0].metadata["type"] == "row"

    def test_no_matching_type_returns_empty(self):
        chunks = self._make_chunks()
        result = filter_by_intent(chunks, ["top_customers"])
        assert result == []

    def test_empty_chunks_returns_empty(self):
        result = filter_by_intent([], ["yearly_summary"])
        assert result == []

    def test_empty_intent_types_returns_empty(self):
        chunks = self._make_chunks()
        result = filter_by_intent(chunks, [])
        assert result == []

    def test_preserves_document_content(self):
        chunks = [_doc("yearly_summary", "content A")]
        result = filter_by_intent(chunks, ["yearly_summary"])
        assert result[0].page_content == "content A"

    def test_partial_overlap(self):
        chunks = self._make_chunks()
        result = filter_by_intent(chunks, ["category_summary", "nonexistent_type"])
        assert len(result) == 1
        assert result[0].metadata["type"] == "category_summary"


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks — vectordb mock tests
# ---------------------------------------------------------------------------

class TestRetrieveRelevantChunks:
    """Tests use a MagicMock vectordb so no real ChromaDB instance is needed."""

    def _mock_db(self, return_docs=None):
        db = MagicMock()
        db.similarity_search.return_value = return_docs or []
        return db

    # --- return value is forwarded as-is -----------------------------------

    def test_returns_list(self):
        db = self._mock_db([_doc("yearly_summary")])
        result = retrieve_relevant_chunks("sales trend", db)
        assert isinstance(result, list)

    def test_returns_docs_from_vectordb(self):
        docs = [_doc("trend_summary"), _doc("yearly_summary")]
        db = self._mock_db(docs)
        result = retrieve_relevant_chunks("trend", db)
        assert result == docs

    def test_returns_empty_when_vectordb_returns_empty(self):
        db = self._mock_db([])
        result = retrieve_relevant_chunks("trend", db)
        assert result == []

    # --- similarity_search called with correct k ---------------------------

    def test_similarity_search_called_with_k20_no_intent(self):
        db = self._mock_db()
        retrieve_relevant_chunks("hello world", db)
        db.similarity_search.assert_called_once_with("hello world", k=20)

    def test_similarity_search_called_with_k20_with_intent(self):
        db = self._mock_db()
        retrieve_relevant_chunks("sales trend", db)
        _, kwargs = db.similarity_search.call_args
        assert kwargs.get("k") == 20

    # --- filter is passed when intent is detected --------------------------

    def test_filter_passed_when_intent_detected(self):
        db = self._mock_db()
        retrieve_relevant_chunks("sales trend", db)
        _, kwargs = db.similarity_search.call_args
        assert "filter" in kwargs

    def test_filter_is_chromadb_where_dict(self):
        """filter must use the {type: {$in: [...]}} ChromaDB format."""
        db = self._mock_db()
        retrieve_relevant_chunks("sales trend", db)
        _, kwargs = db.similarity_search.call_args
        f = kwargs["filter"]
        assert "type" in f
        assert "$in" in f["type"]
        assert isinstance(f["type"]["$in"], list)

    def test_filter_contains_expected_types_for_trend(self):
        db = self._mock_db()
        retrieve_relevant_chunks("sales trend", db)
        _, kwargs = db.similarity_search.call_args
        types_in_filter = kwargs["filter"]["type"]["$in"]
        assert "trend_summary" in types_in_filter

    def test_filter_contains_expected_types_for_city(self):
        db = self._mock_db()
        retrieve_relevant_chunks("sales in each city", db)
        _, kwargs = db.similarity_search.call_args
        types_in_filter = kwargs["filter"]["type"]["$in"]
        assert "city_summary" in types_in_filter

    # --- no filter when intent is None ------------------------------------

    def test_no_filter_when_intent_is_none(self):
        db = self._mock_db()
        retrieve_relevant_chunks("hello world", db)
        _, kwargs = db.similarity_search.call_args
        assert "filter" not in kwargs

    # --- query string forwarded verbatim ----------------------------------

    def test_query_forwarded_to_similarity_search(self):
        db = self._mock_db()
        retrieve_relevant_chunks("trend over time", db)
        args, _ = db.similarity_search.call_args
        assert args[0] == "trend over time"

    def test_query_forwarded_without_intent(self):
        db = self._mock_db()
        retrieve_relevant_chunks("hello world", db)
        args, _ = db.similarity_search.call_args
        assert args[0] == "hello world"
