# Instamart Customer Intelligence & Support Copilot — CAPSTONE PROJECT
# This is an assignment project of AIML by Arun Nivas


: Overview, Reproduce (numbered shell commands, from python -m venv to the last script), Part A, Part B, Part C, Part D, Limitations and Next Steps

## Learning outcomes
Students learn to (1) fetch/read CSV exports and audit data, (2) clean and persist trusted tables in SQLite, (3) write SQL and pandas analytics, (4) segment customers and train a leakage-aware lapse model, (5) redact PII and triage text with retries/output validation, and (6) retrieve policy passages and execute bounded database-backed tools.



Overview:

## Part A — Data Foundation: Clean, Store, Query 

1.	Reproducible setup 

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

Task 2: Audit the raw data 

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

Task3: City counts:
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

#4. Clean orders 

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



orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

orders["category"] = orders["product_id"].map(
    prod.set_index("product_id")["category"]
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