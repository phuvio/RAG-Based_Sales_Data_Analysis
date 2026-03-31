import pandas as pd

df = pd.read_csv('../data/cleaned_superstore.csv')

def create_row_docs(df):
    row_docs = []

    for row in df.itertuples(index=False):
        date_str = row.Order_Date.strftime("%Y-%m-%d")
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        discount = round(row.Discount, 2)

        text = (
            f"Order on {date_str}: Customer {row['Customer Name']} "
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

    for _, row in grouped.itertuples(index=False):
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

    for _, row in grouped.itertuples(index=False):
        sales = round(row.Sales, 2)
        profit = round(row.Profit, 2)
        quantity = int(row.Quantity)
        discount = round(row.Discount, 2)
        subcategory = row._2

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

    for _, row in grouped.itertuples(index=False):
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

    for _, row in grouped.itertuples(index=False):
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

    for _, row in grouped.itertuples(index=False):
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

documents = []
documents += create_row_docs(df)
documents += create_yearly_docs(df)
documents += create_monthly_docs(df)
documents += create_trend_summary_doc(df)
documents += create_top_months_doc(df)
documents += create_category_docs(df)
documents += create_subcategory_docs(df)
documents += create_top_category_doc(df)
documents += create_top_subcategory_doc(df)
documents += create_discount_category_doc(df)
documents += create_region_docs(df)
documents += create_state_docs(df)
documents += create_city_docs(df)
documents += create_top_region_doc(df)
documents += create_category_comparison_doc(df)
documents += create_region_comparison_doc(df)
