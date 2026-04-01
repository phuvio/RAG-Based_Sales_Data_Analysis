import pandas as pd


def _ensure_order_date_datetime(df):
    if not pd.api.types.is_datetime64_any_dtype(df["Order Date"]):
        df = df.copy()
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df


def create_row_docs(df):
    df = _ensure_order_date_datetime(df)
    row_docs = []

    for _, row in df.iterrows():
        date_str = row["Order Date"].strftime("%Y-%m-%d")
        sales = round(row["Sales"], 2)
        profit = round(row["Profit"], 2)
        discount = round(row["Discount"], 2)

        text = (
            f"Order on {date_str}: Customer {row['Customer ID']} "
            f"({row['Segment']}) purchased {row['Quantity']} units of "
            f"{row['Product Name']} (Category: {row['Category']}, Sub-Category: {row['Sub-Category']}) "
            f"in {row['City']}, {row['Region']}. "
            f"Sales: {sales}€, Profit: {profit}€, Discount: {discount}."
        )
        
        metadata = {
            "type": "row",
            "region": row["Region"],
            "category": row["Category"],
            "sub_category": row["Sub-Category"],
            "year": row["Order Date"].year,
            "segment": row["Segment"]
        }
        
        row_docs.append({"text": text, "metadata": metadata})
    return row_docs

def create_yearly_docs(df):
    df = _ensure_order_date_datetime(df)
    docs = []
    
    yearly = df.groupby(df["Order Date"].dt.year).agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Discount": "mean"
    }).reset_index()

    yearly.rename(columns={"Order Date": "Year"}, inplace=True)

    for row in yearly.itertuples(index=False):
        year = int(row.Year)
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        quantity = int(row.Quantity)
        discount = round(row.Discount, 2)

        text = (
            f"In {year}, total sales were {sales}€, total profit was {profit}€, "
            f"and total quantity sold was {quantity}. "
            f"The average discount was {discount}."
        )

        metadata = {
            "type": "yearly_summary",
            "year": year
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_monthly_docs(df):
    df = _ensure_order_date_datetime(df)
    docs = []
    
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month

    monthly = df.groupby(["Year", "Month"]).agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()

    for row in monthly.itertuples(index=False):
        year = int(row.Year)
        month = int(row.Month)
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        quantity = int(row.Quantity)

        text = (
            f"In {year}-{month:02d}, total sales were {sales}€, "
            f"total profit was {profit}€, and total quantity sold was {quantity}."
        )

        metadata = {
            "type": "monthly_summary",
            "year": year,
            "month": month
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_trend_summary_doc(df):
    df = _ensure_order_date_datetime(df)
    docs = []
    
    yearly = df.groupby(df["Order Date"].dt.year)["Sales"].sum().reset_index()
    yearly.rename(columns={"Order Date": "Year"}, inplace=True)

    trend = "increasing" if yearly["Sales"].iloc[-1] > yearly["Sales"].iloc[0] else "decreasing"

    text = (
        f"Over the observed years, sales show an overall {trend} trend. "
        f"Sales increased from {round(yearly['Sales'].iloc[0],2)}€ "
        f"to {round(yearly['Sales'].iloc[-1],2)}€."
    )

    metadata = {
        "type": "trend_summary"
    }

    docs.append({"text": text, "metadata": metadata})

    return docs

def create_top_months_doc(df):
    df = _ensure_order_date_datetime(df)
    docs = []
    
    df["YearMonth"] = df["Order Date"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index()

    top = monthly.sort_values("Sales", ascending=False).head(3)

    text = "Top 3 months by sales are: " + ", ".join(
        [f"{str(row['YearMonth'])} ({round(row['Sales'],2)}€)" for _, row in top.iterrows()]
    )

    docs.append({
        "text": text,
        "metadata": {"type": "top_months"}
    })

    return docs

def create_category_docs(df):
    docs = []
    
    grouped = df.groupby("Category").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Discount": "mean"
    }).reset_index()

    for row in grouped.itertuples(index=False):
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        quantity = int(row.Quantity)
        discount = round(row.Discount, 2)

        text = (
            f"Category {row.Category} generated total sales of {sales}€, "
            f"total profit of {profit}€, and total quantity sold of {quantity}. "
            f"The average discount was {discount}."
        )

        metadata = {
            "type": "category_summary",
            "category": row.Category
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_subcategory_docs(df):
    docs = []
    
    grouped = df.groupby(["Category", "Sub-Category"]).agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Discount": "mean"
    }).reset_index()

    for row in grouped.itertuples(index=False):
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        quantity = int(row.Quantity)
        discount = round(row.Discount, 2)
        subcategory = row[1]

        text = (
            f"Sub-category {subcategory} in category {row.Category} "
            f"generated total sales of {sales}€, total profit of {profit}€, "
            f"and total quantity sold of {quantity}. "
            f"The average discount was {discount}."
        )

        metadata = {
            "type": "subcategory_summary",
            "category": row.Category,
            "sub_category": subcategory
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_top_category_doc(df):
    docs = []
    
    grouped = df.groupby("Category")["Sales"].sum().reset_index()
    top = grouped.sort_values("Sales", ascending=False).iloc[0]

    text = (
        f"The category with the highest total sales is {top['Category']} "
        f"with {round(top['Sales'], 2)}€ in revenue."
    )

    docs.append({
        "text": text,
        "metadata": {"type": "top_category"}
    })

    return docs

def create_top_subcategory_doc(df):
    docs = []
    
    grouped = df.groupby(["Category", "Sub-Category"])["Profit"].sum().reset_index()
    top = grouped.sort_values("Profit", ascending=False).iloc[0]

    text = (
        f"The most profitable sub-category is {top['Sub-Category']} "
        f"in category {top['Category']} with total profit of {round(top['Profit'], 2)}€."
    )

    docs.append({
        "text": text,
        "metadata": {"type": "top_subcategory"}
    })

    return docs

def create_discount_category_doc(df):
    docs = []
    
    grouped = df.groupby("Category")["Discount"].mean().reset_index()
    top = grouped.sort_values("Discount", ascending=False).iloc[0]

    text = (
        f"The category with the highest average discount is {top['Category']} "
        f"with an average discount of {round(top['Discount'], 2)}."
    )

    docs.append({
        "text": text,
        "metadata": {"type": "discount_category"}
    })

    return docs

def create_region_docs(df):
    docs = []
    
    grouped = df.groupby("Region").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()

    for row in grouped.itertuples(index=False):
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        quantity = int(row.Quantity)
        region = row.Region

        text = (
            f"Region {region} generated total sales of {sales}€, "
            f"total profit of {profit}€, and total quantity sold of {quantity}."
        )

        metadata = {
            "type": "region_summary",
            "region": region
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_state_docs(df):
    docs = []
    
    grouped = df.groupby("State").agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    for row in grouped.itertuples(index=False):
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)

        text = (
            f"In state {row.State}, total sales were {sales}€ "
            f"and total profit was {profit}€."
        )

        metadata = {
            "type": "state_summary",
            "state": row.State
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_city_docs(df):
    docs = []
    
    grouped = df.groupby("City").agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    for row in grouped.itertuples(index=False):
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)

        text = (
            f"In city {row.City}, total sales were {sales}€ "
            f"and total profit was {profit}€."
        )

        metadata = {
            "type": "city_summary",
            "city": row.City
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_top_region_doc(df):
    docs = []
    
    grouped = df.groupby("Region")["Sales"].sum().reset_index()
    top = grouped.sort_values("Sales", ascending=False).iloc[0]

    text = (
        f"The region with the highest total sales is {top['Region']} "
        f"with {round(top['Sales'], 2)}€ in revenue."
    )

    docs.append({
        "text": text,
        "metadata": {"type": "top_region"}
    })

    return docs

def create_category_comparison_doc(df):
    docs = []
    
    grouped = df.groupby("Category")["Sales"].sum().reset_index()

    if "Technology" in grouped["Category"].values and "Furniture" in grouped["Category"].values:
        tech = grouped[grouped["Category"] == "Technology"]["Sales"].values[0]
        furn = grouped[grouped["Category"] == "Furniture"]["Sales"].values[0]

        text = (
            f"Technology sales total {round(tech,2)}€, while Furniture sales total {round(furn,2)}€. "
            f"Technology performs {'better' if tech > furn else 'worse'} than Furniture."
        )

        docs.append({
            "text": text,
            "metadata": {"type": "category_comparison"}
        })

    return docs

def create_region_comparison_doc(df):
    docs = []
    
    grouped = df.groupby("Region")["Profit"].sum().reset_index()

    if "West" in grouped["Region"].values and "East" in grouped["Region"].values:
        west = grouped[grouped["Region"] == "West"]["Profit"].values[0]
        east = grouped[grouped["Region"] == "East"]["Profit"].values[0]

        text = (
            f"West region profit is {round(west,2)}€, while East region profit is {round(east,2)}€. "
            f"West performs {'better' if west > east else 'worse'} than East in terms of profit."
        )

        docs.append({
            "text": text,
            "metadata": {"type": "region_comparison"}
        })

    return docs

def create_segment_docs(df):
    docs = []
    
    grouped = df.groupby("Segment").agg({
        "Sales": "sum",
        "Profit": "sum",
        "Quantity": "sum",
        "Discount": "mean"
    }).reset_index()

    for _, row in grouped.iterrows():
        text = (
            f"Segment {row['Segment']} generated total sales of {round(row['Sales'],2)}€, "
            f"total profit of {round(row['Profit'],2)}€, and sold {int(row['Quantity'])} units. "
            f"The average discount was {round(row['Discount'],2)}."
        )

        metadata = {
            "type": "segment_summary",
            "segment": row["Segment"]
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_category_trend_docs(df):
    docs = []
    
    df["Year"] = df["Order Date"].dt.year

    grouped = df.groupby(["Category", "Year"]).agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    for _, row in grouped.iterrows():
        text = (
            f"In {int(row['Year'])}, category {row['Category']} generated "
            f"{round(row['Sales'],2)}€ in sales and {round(row['Profit'],2)}€ in profit."
        )

        metadata = {
            "type": "category_trend",
            "category": row["Category"],
            "year": int(row["Year"])
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_region_category_docs(df):
    docs = []
    
    grouped = df.groupby(["Region", "Category"]).agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    for _, row in grouped.iterrows():
        text = (
            f"In the {row['Region']} region, category {row['Category']} generated "
            f"{round(row['Sales'],2)}€ in sales and {round(row['Profit'],2)}€ in profit."
        )

        metadata = {
            "type": "region_category_summary",
            "region": row["Region"],
            "category": row["Category"]
        }

        docs.append({"text": text, "metadata": metadata})

    return docs

def create_top_customers_doc(df, top_n=5):
    docs = []
    
    grouped = df.groupby("Customer Name").agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()

    top = grouped.sort_values("Sales", ascending=False).head(top_n)

    text = "Top customers by sales are: " + ", ".join(
        [
            f"{row['Customer Name']} ({round(row['Sales'],2)}€)"
            for _, row in top.iterrows()
        ]
    )

    docs.append({
        "text": text,
        "metadata": {"type": "top_customers"}
    })

    return docs

def create_segment_comparison_doc(df):
    docs = []
    
    grouped = df.groupby("Segment")["Sales"].sum().reset_index()

    if len(grouped) >= 2:
        sorted_segments = grouped.sort_values("Sales", ascending=False)

        best = sorted_segments.iloc[0]
        worst = sorted_segments.iloc[-1]

        text = (
            f"The best performing segment is {best['Segment']} with {round(best['Sales'],2)}€ in sales, "
            f"while the lowest is {worst['Segment']} with {round(worst['Sales'],2)}€."
        )

        docs.append({
            "text": text,
            "metadata": {"type": "segment_comparison"}
        })

    return docs
