"""
Unit tests for src/generate_documents.py

Covers:
- Document creation functions for rows, monthly/yearly summaries, trends, categories, regions, etc.
- New document creation functions for statistics, product rankings, and distributions
- Metadata structure validation
- Date handling and formatting
- Data aggregation and statistics calculations
"""

import sys
from pathlib import Path
import pandas as pd
import pytest
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generate_documents


@pytest.fixture
def sample_df():
    """Create a sample dataframe matching superstore structure."""
    return pd.DataFrame({
        "Order Date": ["2023-01-15", "2023-02-20", "2023-03-10", "2024-01-05", "2024-02-14"],
        "Customer ID": ["C001", "C002", "C001", "C003", "C002"],
        "Segment": ["Consumer", "Corporate", "Consumer", "Home Office", "Consumer"],
        "Quantity": [2, 5, 1, 3, 4],
        "Product Name": ["Pen", "Desk", "Pen", "Chair", "Desk"],
        "Category": ["Office Supplies", "Furniture", "Office Supplies", "Furniture", "Furniture"],
        "Sub-Category": ["Pens", "Desks", "Pens", "Chairs", "Desks"],
        "City": ["New York", "Los Angeles", "New York", "Chicago", "Los Angeles"],
        "State": ["NY", "CA", "NY", "IL", "CA"],
        "Region": ["East", "West", "East", "Central", "West"],
        "Sales": [100.0, 500.0, 50.0, 300.0, 450.0],
        "Profit": [20.0, 100.0, 10.0, 60.0, 90.0],
        "Discount": [0.0, 0.1, 0.0, 0.05, 0.1],
    })


@pytest.fixture
def sample_df_datetime(sample_df):
    """Create sample dataframe with datetime Order Date."""
    df = sample_df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df


class TestEnsureOrderDateDatetime:
    def test_converts_string_dates_to_datetime(self):
        df = pd.DataFrame({"Order Date": ["2023-01-15", "2023-01-20"]})
        result = generate_documents._ensure_order_date_datetime(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result["Order Date"])
        assert result["Order Date"].dtype == 'datetime64[ns]'

    def test_preserves_existing_datetime(self, sample_df_datetime):
        result = generate_documents._ensure_order_date_datetime(sample_df_datetime)
        
        assert pd.api.types.is_datetime64_any_dtype(result["Order Date"])

    def test_does_not_modify_original_string_df(self):
        df = pd.DataFrame({"Order Date": ["2023-01-15"]})
        original_dtype = df["Order Date"].dtype
        generate_documents._ensure_order_date_datetime(df)
        
        assert df["Order Date"].dtype == original_dtype


class TestCreateRowDocs:
    def test_creates_documents_for_each_row(self, sample_df):
        docs = generate_documents.create_row_docs(sample_df)
        
        assert len(docs) == len(sample_df)

    def test_document_has_text_and_metadata(self, sample_df):
        docs = generate_documents.create_row_docs(sample_df)
        
        for doc in docs:
            assert "text" in doc
            assert "metadata" in doc
            assert isinstance(doc["text"], str)
            assert isinstance(doc["metadata"], dict)

    def test_metadata_contains_required_fields(self, sample_df):
        docs = generate_documents.create_row_docs(sample_df)
        
        for doc in docs:
            metadata = doc["metadata"]
            assert metadata["type"] == "row"
            assert "region" in metadata
            assert "category" in metadata
            assert "sub_category" in metadata
            assert "year" in metadata
            assert "segment" in metadata

    def test_text_contains_customer_and_product_info(self, sample_df):
        docs = generate_documents.create_row_docs(sample_df)
        
        assert "C001" in docs[0]["text"]
        assert "Pen" in docs[0]["text"]
        assert "Office Supplies" in docs[0]["text"]

    def test_rounds_sales_profit_discount(self, sample_df):
        docs = generate_documents.create_row_docs(sample_df)
        
        assert "100.0€" in docs[0]["text"]
        assert "20.0€" in docs[0]["text"]


class TestCreateYearlyDocs:
    def test_creates_yearly_summaries(self, sample_df):
        docs = generate_documents.create_yearly_docs(sample_df)
        
        assert len(docs) == 2  # 2023 and 2024

    def test_yearly_metadata_structure(self, sample_df):
        docs = generate_documents.create_yearly_docs(sample_df)
        
        for doc in docs:
            assert doc["metadata"]["type"] == "yearly_summary"
            assert "year" in doc["metadata"]
            assert isinstance(doc["metadata"]["year"], int)

    def test_yearly_text_includes_totals(self, sample_df):
        docs = generate_documents.create_yearly_docs(sample_df)
        
        for doc in docs:
            assert "sales were" in doc["text"]
            assert "profit was" in doc["text"]
            assert "€" in doc["text"]

    def test_yearly_aggregation_correct(self, sample_df):
        docs = generate_documents.create_yearly_docs(sample_df)
        
        # 2023: Sales = 100 + 500 + 50 = 650, Profit = 20 + 100 + 10 = 130
        doc_2023 = [d for d in docs if d["metadata"]["year"] == 2023][0]
        assert "650" in doc_2023["text"]
        assert "130" in doc_2023["text"]


class TestCreateMonthlyDocs:
    def test_creates_monthly_summaries(self, sample_df):
        docs = generate_documents.create_monthly_docs(sample_df)
        
        assert len(docs) == 5  # 5 distinct month-year combinations

    def test_monthly_metadata_structure(self, sample_df):
        docs = generate_documents.create_monthly_docs(sample_df)
        
        for doc in docs:
            assert doc["metadata"]["type"] == "monthly_summary"
            assert "year" in doc["metadata"]
            assert "month" in doc["metadata"]

    def test_monthly_text_format(self, sample_df):
        docs = generate_documents.create_monthly_docs(sample_df)
        
        for doc in docs:
            text = doc["text"]
            assert "sales were" in text
            assert "profit was" in text
            assert "€" in text


class TestCreateTrendSummaryDoc:
    def test_creates_trend_summary(self, sample_df):
        docs = generate_documents.create_trend_summary_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "trend_summary"

    def test_trend_text_includes_sales_values(self, sample_df):
        docs = generate_documents.create_trend_summary_doc(sample_df)
        
        text = docs[0]["text"]
        assert "€" in text
        assert "trend" in text.lower()

    def test_trend_detection_increasing(self):
        df = pd.DataFrame({
            "Order Date": ["2023-01-01", "2024-01-01"],
            "Sales": [100.0, 200.0],
        })
        docs = generate_documents.create_trend_summary_doc(df)
        
        assert "increasing" in docs[0]["text"]

    def test_trend_detection_decreasing(self):
        df = pd.DataFrame({
            "Order Date": ["2023-01-01", "2024-01-01"],
            "Sales": [200.0, 100.0],
        })
        docs = generate_documents.create_trend_summary_doc(df)
        
        assert "decreasing" in docs[0]["text"]


class TestCreateTopMonthsDoc:
    def test_creates_top_months_doc(self, sample_df):
        docs = generate_documents.create_top_months_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "top_months"

    def test_top_months_text_format(self, sample_df):
        docs = generate_documents.create_top_months_doc(sample_df)
        
        text = docs[0]["text"]
        assert "Top 3 months" in text or "top" in text.lower()
        assert "€" in text


class TestCreateCategoryDocs:
    def test_creates_category_summaries(self, sample_df):
        docs = generate_documents.create_category_docs(sample_df)
        
        assert len(docs) == 2  # Office Supplies and Furniture

    def test_category_metadata_structure(self, sample_df):
        docs = generate_documents.create_category_docs(sample_df)
        
        for doc in docs:
            assert doc["metadata"]["type"] == "category_summary"
            assert "category" in doc["metadata"]

    def test_category_text_includes_stats(self, sample_df):
        docs = generate_documents.create_category_docs(sample_df)
        
        for doc in docs:
            text = doc["text"]
            assert "sales of" in text
            assert "profit of" in text
            assert "€" in text


class TestCreateSubcategoryDocs:
    def test_creates_subcategory_summaries(self, sample_df):
        docs = generate_documents.create_subcategory_docs(sample_df)
        
        assert len(docs) == 3  # Pens, Desks, Chairs

    def test_subcategory_metadata_structure(self, sample_df):
        docs = generate_documents.create_subcategory_docs(sample_df)
        
        for doc in docs:
            assert doc["metadata"]["type"] == "subcategory_summary"
            assert "category" in doc["metadata"]
            assert "sub_category" in doc["metadata"]


class TestCreateTopCategoryDoc:
    def test_creates_top_category_doc(self, sample_df):
        docs = generate_documents.create_top_category_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "top_category"

    def test_top_category_identifies_highest_sales(self, sample_df):
        docs = generate_documents.create_top_category_doc(sample_df)
        
        text = docs[0]["text"]
        assert "Furniture" in text  # Furniture has higher sales


class TestCreateTopSubcategoryDoc:
    def test_creates_top_subcategory_doc(self, sample_df):
        docs = generate_documents.create_top_subcategory_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "top_subcategory"

    def test_top_subcategory_identifies_highest_profit(self, sample_df):
        docs = generate_documents.create_top_subcategory_doc(sample_df)
        
        text = docs[0]["text"]
        assert "€" in text


class TestCreateDiscountCategoryDoc:
    def test_creates_discount_category_doc(self, sample_df):
        docs = generate_documents.create_discount_category_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "discount_category"


class TestCreateRegionDocs:
    def test_creates_region_summaries(self, sample_df):
        docs = generate_documents.create_region_docs(sample_df)
        
        assert len(docs) == 3  # East, West, Central

    def test_region_metadata_structure(self, sample_df):
        docs = generate_documents.create_region_docs(sample_df)
        
        for doc in docs:
            assert doc["metadata"]["type"] == "region_summary"
            assert "region" in doc["metadata"]

    def test_region_text_includes_stats(self, sample_df):
        docs = generate_documents.create_region_docs(sample_df)
        
        for doc in docs:
            text = doc["text"]
            assert "sales of" in text
            assert "€" in text


class TestCreateOverallStatsDoc:
    """Tests for new create_overall_stats_doc function."""
    
    def test_creates_overall_stats_doc(self, sample_df):
        docs = generate_documents.create_overall_stats_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "overall_statistics"

    def test_overall_stats_text_format(self, sample_df):
        docs = generate_documents.create_overall_stats_doc(sample_df)
        
        text = docs[0]["text"]
        assert "Average sales per order" in text
        assert "median sales" in text
        assert "Average profit" in text
        assert "median profit" in text
        assert "€" in text

    def test_overall_stats_calculations(self, sample_df):
        docs = generate_documents.create_overall_stats_doc(sample_df)
        
        mean_sales = round(sample_df["Sales"].mean(), 2)
        median_sales = round(sample_df["Sales"].median(), 2)
        
        text = docs[0]["text"]
        assert str(mean_sales) in text
        assert str(median_sales) in text


class TestCreateCategoryStatsDoc:
    """Tests for new create_category_stats_docs function."""
    
    def test_creates_category_stats_docs(self, sample_df):
        docs = generate_documents.create_category_stats_docs(sample_df)
        
        assert len(docs) == 2  # Office Supplies and Furniture

    def test_category_stats_metadata_structure(self, sample_df):
        docs = generate_documents.create_category_stats_docs(sample_df)
        
        for doc in docs:
            assert doc["metadata"]["type"] == "category_statistics"
            assert "category" in doc["metadata"]

    def test_category_stats_text_includes_means_medians(self, sample_df):
        docs = generate_documents.create_category_stats_docs(sample_df)
        
        for doc in docs:
            text = doc["text"]
            assert "average sales" in text.lower()
            assert "median sales" in text.lower()
            assert "average profit" in text.lower()
            assert "median profit" in text.lower()


class TestCreateTopBottomProducts:
    """Tests for new create_top_bottom_docs function."""
    
    def test_creates_top_bottom_docs(self, sample_df):
        docs = generate_documents.create_top_bottom_docs(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "top_bottom_products"

    def test_top_bottom_text_format(self, sample_df):
        docs = generate_documents.create_top_bottom_docs(sample_df)
        
        text = docs[0]["text"]
        assert "Top 3 products" in text
        assert "Lowest performing products" in text
        assert "€" in text

    def test_top_bottom_identifies_correct_products(self, sample_df):
        docs = generate_documents.create_top_bottom_docs(sample_df)
        
        text = docs[0]["text"]
        # Desk has highest profit, Pen has lowest profit
        assert "Desk" in text


class TestCreateDistributionDoc:
    """Tests for new create_distribution_doc function."""
    
    def test_creates_distribution_doc(self, sample_df):
        docs = generate_documents.create_distribution_doc(sample_df)
        
        assert len(docs) == 1
        assert docs[0]["metadata"]["type"] == "distribution_summary"

    def test_distribution_text_format(self, sample_df):
        docs = generate_documents.create_distribution_doc(sample_df)
        
        text = docs[0]["text"]
        assert "Sales distribution" in text
        assert "minimum" in text
        assert "maximum" in text
        assert "standard deviation" in text
        assert "€" in text

    def test_distribution_calculations(self, sample_df):
        docs = generate_documents.create_distribution_doc(sample_df)
        
        min_sales = round(sample_df["Sales"].min(), 2)
        max_sales = round(sample_df["Sales"].max(), 2)
        
        text = docs[0]["text"]
        assert str(min_sales) in text
        assert str(max_sales) in text


class TestGenerateAllDocuments:
    def test_generate_all_documents_includes_new_functions(self, sample_df):
        docs = generate_documents.generate_all_documents(sample_df)
        
        # Check that new document types are included
        doc_types = {doc["metadata"].get("type") for doc in docs}
        
        assert "overall_statistics" in doc_types
        assert "category_statistics" in doc_types
        assert "top_bottom_products" in doc_types
        assert "distribution_summary" in doc_types
        assert "row" in doc_types  # Original functions still included

    def test_generate_all_documents_returns_list(self, sample_df):
        docs = generate_documents.generate_all_documents(sample_df)
        
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_all_documents_have_required_fields(self, sample_df):
        docs = generate_documents.generate_all_documents(sample_df)
        
        for doc in docs:
            assert "text" in doc
            assert "metadata" in doc
            assert "type" in doc["metadata"]
            assert isinstance(doc["text"], str)
            assert len(doc["text"]) > 0
