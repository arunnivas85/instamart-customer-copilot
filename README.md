# Overview:
Instamart Customer Intelligence & Support Copilot — CAPSTONE PROJECT. This is an assignment project of AIML by Arun Nivas


## Part A — Data Foundation: Clean, Store, Query 

# Task 1.	Reproducible setup 

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python run_project.py
```

-------------------------------------

# Task 2: Audit the raw data 

```bash
 python data_pipeline.py > data_pipeline_output.txt
 ```


orders.csv
Raw rows:	13084
Exact duplicate rows:	119
Rows with status == "pending":	656
Distinct order_id values:	12309

order_amount entries that cannot be converted to a number: 120
Negative delivery_minutes: 106
Blank delivery_minutes: 2541

customers.csv:
Blank age: 61
age values outside 10–100: 6
Distinct raw city spellings: 24
Distinct non-blank raw gender spellings: 10
Blank gender: 30

--------------------------------------

# Task 3: City counts:
Pune         227
Delhi        210
Bengaluru    197
Chennai      191
Hyderabad    191
Mumbai       184

Gender counts:
female       610
male         560
unknown       30

Missing ages: 0
Age range: 18–60
Signup dates: all values parse as dates


------------------------------------------------

# Task 4: Clean orders 

```bash
# i. Convert order_ts to datetime

orders["order_ts"] = pd.to_datetime(orders["order_ts"])


# ii. Remove exact duplicate rows
orders = orders.drop_duplicates()

## iii. Keep the latest record for each order_id
orders = (
    orders.sort_values(["order_id", "order_ts"])
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# iv. Convert order_amount to numeric and fill missing values

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

price_by_product = prod.set_index("product_id")["price_inr"]
missing = orders["order_amount"].isna()

orders.loc[missing, "order_amount"] = (
    orders.loc[missing, "product_id"].map(price_by_product)
    * orders.loc[missing, "quantity"]
    * (1 - orders.loc[missing, "discount_pct"] / 100)
).round(2)


# v.Set negative delivery_minutes to missing. 
 12309 rows (one per order_id); 
 11110 completed and 1199 cancelled; 
 0 pending rows; 
 0 missing order_amount; 
 0 missing delivery_minutes among completed orders and 1199 missing among cancelled orders; 
 The sum of order_amount over completed orders is  INR 2998668.80


# Standardize customer fields.
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# Recalculate missing order amounts from product price.
    price_by_product = products.set_index("product_id")["price_inr"]
    missing_amount = orders["order_amount"].isna()
    orders.loc[missing_amount, "order_amount"] = (
        orders.loc[missing_amount, "product_id"].map(price_by_product)
        * orders.loc[missing_amount, "quantity"]
        * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)

# Negative delivery times are invalid. Keep cancelled orders missing.
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    cancelled = orders["status"].eq("cancelled")
    orders.loc[cancelled, "delivery_minutes"] = np.nan

# Fill missing delivery times only for completed orders, using the
# completed-order median for each product category.
    orders["category"] = orders["product_id"].map(
        products.set_index("product_id")["category"]
    )
    completed = orders["status"].eq("completed")
    completed_orders = orders.loc[completed]
    category_median = completed_orders.groupby("category")[
        "delivery_minutes"
    ].transform("median")
    orders.loc[completed, "delivery_minutes"] = (
        completed_orders["delivery_minutes"].fillna(category_median)
    )

```


----------------------------------------------------

# Task 5 : Store and validate

```bash
# Assumes customers, products, and orders are already cleaned DataFrames.
from pathlib import Path
import sqlite3

import pandas as pd

clean_dir = Path("data/clean")
clean_dir.mkdir(parents=True, exist_ok=True)
db_path = Path("data/instamart.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

# Store order_ts as text in YYYY-MM-DD HH:MM:SS format.
orders["order_ts"] = pd.to_datetime(orders["order_ts"]).dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

tables = {
    "customers": customers,
    "products": products,
    "orders": orders,
}

for name, df in tables.items():
    df.to_csv(clean_dir / f"{name}.csv", index=False)

with sqlite3.connect(db_path) as connection:
    for name, df in tables.items():
        df.to_sql(name, connection, if_exists="replace", index=False)

# Reference validation checkpoints
completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

assert customers["city"].nunique() == 6
assert customers["gender"].isin(["male", "female", "unknown"]).all()
assert customers["age"].notna().all()
assert customers["age"].between(18, 60).all()
assert len(orders) == 12_309
assert orders["order_id"].is_unique
assert completed.sum() == 11_110
assert cancelled.sum() == 1_199
assert not orders["status"].eq("pending").any()
assert orders["order_amount"].notna().all()
assert orders.loc[completed, "delivery_minutes"].notna().all()
assert orders.loc[cancelled, "delivery_minutes"].isna().all()
assert round(orders.loc[completed, "order_amount"].sum(), 2) == 2_998_668.80

print("All validation checkpoints passed.")

```
----------------------------------------------------------


# Task 6: SQL analysis 

Queries are stored in sql/queries.sql


# Task 7: Pandas cross-check and city-level exploration

The pivot table reproduces Q2's completed-order revenue by city and product
category. Its `All` row and column include the grand total; the assertion
cross-checks that total against the equivalent SQL aggregation.

```bash
from pathlib import Path
import sqlite3

import pandas as pd

db_path = Path("data") / "instamart.db"
with sqlite3.connect(db_path) as connection:
    completed_sales = pd.read_sql_query(
        """
        SELECT c.city, p.category, o.order_amount
        FROM orders o
        JOIN customers c USING (customer_id)
        JOIN products p USING (product_id)
        WHERE o.status = 'completed'
        """,
        connection,
    )
    sql_total = pd.read_sql_query(
        """
        SELECT ROUND(SUM(o.order_amount), 2) AS revenue
        FROM orders o
        JOIN customers c USING (customer_id)
        JOIN products p USING (product_id)
        WHERE o.status = 'completed'
        """,
        connection,
    ).loc[0, "revenue"]

revenue_by_city_category = pd.pivot_table(
    completed_sales,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All",
)
pivot_total = revenue_by_city_category.loc["All", "All"]
assert round(pivot_total, 2) == sql_total

city_order_amount = completed_sales.groupby("city").agg(
    sum_order_amount=("order_amount", "sum"),
    mean_order_amount=("order_amount", "mean"),
    order_count=("order_amount", "count"),
)
city_order_amount["sum_order_amount"] = city_order_amount[
    "sum_order_amount"
].round(2)
city_order_amount["mean_order_amount"] = city_order_amount[
    "mean_order_amount"
].round(2)

print(revenue_by_city_category)
print(f"SQL total: INR {sql_total:,.2f}")
print(f"Pivot grand total: INR {pivot_total:,.2f}")
print("Totals match:", round(pivot_total, 2) == sql_total)
print(city_order_amount)
```

The computed SQL and pivot grand totals both equal INR 2998668.80.
The city aggregation reports the following completed-order statistics:

| City | Sum of order_amount (INR) | Mean order_amount (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### PART A ENDS ###


