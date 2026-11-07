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


## PART B BEGINS ##

### Reproduce Part B

From the repository root, run:

```powershell
python run_test.py
```

The script uses `data/clean/orders.csv`, `data/clean/customers.csv`, and
`data/clean/products.csv`. It uses fixed seeds for K-means, the stratified
split, the random forest, and the oversampling step. It writes:

- `outputs/customer_segments.csv`: `customer_id`, `recency`, `frequency`,
  `monetary`, `segment_id`, `segment_name`, `is_premium`.
- `outputs/lapse_model.joblib`: the selected fitted preprocessing/model
  pipeline.
- `outputs/retention_list.csv`: `customer_id`, `lapse_probability`,
  `actually_lapsed`.

### 1. RFM table

The RFM table contains 1,175 customers with at least one completed order,
measured at 2025-10-01 00:00:00. Recency has median 19 days, minimum 0 and
maximum 180; frequency averages 9.46 orders and peaks at 33; monetary value
averages INR 2,552.06 and peaks at INR 14,766.55. Recency is whole elapsed days,
frequency counts distinct completed order IDs, and monetary is the sum of
completed `order_amount`.

### 2. K-means selection

RFM columns are standardized together before fitting each model.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

The sharpest elbow is at K=4, and K=4 also has the highest silhouette score in
the business-approved choices {3, 4, 5}. Four groups give the retention team
distinct premium, frequent, cooling-off and dormant campaign audiences without
adding less interpretable subdivisions. Inertia at K=8 is necessarily no
higher than at K=2 because adding centroids can only maintain or reduce the
within-cluster sum of squared distances; that improvement alone does not
account for added complexity or business usefulness, so it cannot determine K.

### 3. Segment profiles and projection

| Segment ID | Name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Campaign action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Cooling Off | 436 | 28.86 | 3.77 | 1,057.62 | Send a low-cost win-back offer and make the next purchase easy. |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Offer VIP service and relevant high-value bundles; this group has the highest mean monetary value. |
| 2 | Frequent Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Use replenishment reminders and loyalty rewards rather than broad discounts. |
| 3 | Dormant | 182 | 109.04 | 2.20 | 685.99 | Run a targeted reactivation campaign and suppress repeated incentives after non-response. |

The Premium flag is true only for the segment with the highest mean monetary
value. Cluster numbers are arbitrary labels, not an ordered ranking of value,
recency, or loyalty. PCA on the scaled RFM matrix explains 0.6343 and 0.2180 of
the variance in its first two components (total 0.8523).

### 4. Lapse cohort, features, and split

The cohort includes every customer with a completed order before 2025-09-01.
A customer is labeled lapsed when they have no completed order from
2025-09-01 through 2025-09-30: 1,156 customers, of whom 443 (38.32%) lapsed.
The stratified 80/20 split contains 924 training and 232 test customers.

Features use only completed orders before the cutoff for recency, distinct-order
frequency, monetary value, mean discount, mean delivery time, and favourite
product category. Cancellation rate uses cancelled orders divided by all
completed or cancelled orders before the cutoff. City, age, gender, and signup
tenure are taken from the clean customer table. The September window is used
only to create the label. No order dated 2025-09-01 or later entered the
features, and Part 3 segment labels were not used as features.

### 5. Leakage-safe preprocessing

The numeric median imputer and `StandardScaler`, and categorical imputer and
`OneHotEncoder(drop="first", handle_unknown="ignore")`, are inside a
scikit-learn `Pipeline`/`ColumnTransformer` and fitted separately inside each
cross-validation training fold; the final fitted preprocessing is learned from
the training split only. Test rows are transformed, never fitted. No encoder,
scaler, or resampler saw test rows. A code assertion verifies that the
transformed train and test feature matrices contain no missing values.

The fitted logistic pipeline's frequency-scaler mean is 8.115 orders, compared
with the full-cohort mean of 8.029. They differ because the scaler is fitted on
the 80% training subset only, whose sample mean need not equal the full-cohort
mean.

### 6. Tuning and test evaluation

The always-not-lapsed baseline has 61.64% accuracy and lapsed-class F1 of 0;
its confusion matrix (actual rows, predicted columns, class order 0 then 1) is
`[[143, 0], [89, 0]]`.

Both searches use five-fold CV with `scoring="f1"` and are fitted on training
data only:

- Logistic regression (`max_iter=1000`), C in {0.01, 0.1, 1, 10, 100}: best
  C=100; best CV F1=0.651.
- Random forest (`n_estimators=100`, `random_state=42`), `max_depth` in {3, 5,
  8} and `min_samples_leaf` in {5, 10, 20}: best depth=8 and leaf size=10;
  best CV F1=0.687 (45 fits).

| Model | Test confusion matrix | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---:|---:|
| Logistic regression | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Random forest | `[[110, 33], [32, 57]]` | 0.633 | 0.640 | 0.637 |

The confusion matrices use actual rows and predicted columns, with class order
0 then 1. Logistic regression is selected: it has the stronger held-out
lapsed-class precision, recall and F1 in this run, and its test F1 exceeds the
0.55 target. Missing a customer who is about to lapse can mean losing future
orders, so recall matters to the retention team. But contacting loyal customers
unnecessarily spends campaign budget and can train them to wait for discounts,
so precision also matters. The scores are a single split's estimates; campaign
thresholds should be chosen with outreach capacity and incentive cost in mind.

### 7. Training-only class-balance experiment

The original logistic model scores the untouched test set at precision 0.644,
recall 0.652 and F1 0.648. The training split has 570 non-lapsed and 354 lapsed
rows; keeping all original rows and adding 216 replacement-sampled lapsed rows
produces 570 of each class. The refitted logistic model scores the same,
unchanged test set at precision 0.589, recall 0.854 and F1 0.697.

Recall moves most (+0.202); precision falls by 0.055 and F1 rises by 0.049.
The balanced model catches more likely lapsers, but also flags more loyal
customers, so the retention team should budget for extra outreach and avoid
automatically applying costly incentives to every flagged customer.

### 8. Interpretation and persisted outputs

The five largest random-forest feature importances are:

| Feature | Importance | Business reading |
|---|---:|---|
| Monetary | 0.280 | Historical spend is the strongest split signal; protect high-value customers. |
| Frequency | 0.278 | Order cadence is nearly as informative; regularity helps distinguish risk. |
| Recency | 0.109 | Time since the last order is an important warning signal. |
| Average delivery minutes | 0.060 | Delivery experience contributes to predicted risk. |
| Cancellation rate | 0.052 | Prior cancelled orders contribute additional predictive information. |

For the selected best logistic regression, the recency coefficient is positive
(+0.039): longer time since the last order is associated with higher lapse
odds, holding other features fixed. Frequency is negative (-1.697): more
completed orders are associated with lower lapse odds. Cancellation rate is
negative (-0.124) in this fitted model; this conditional association is
counter-intuitive and should be investigated rather than interpreted as causal.

The selected pipeline is saved to `outputs/lapse_model.joblib`, reloaded with
`joblib.load`, and asserted to reproduce every original test prediction exactly.
Among test customers marked Premium by the whole-window RFM segments, the top
10 lapse probabilities are saved to `outputs/retention_list.csv`; 3 of the 10
actually lapsed. The retention list is illustrative and is not a leakage-free
evaluation because its Premium filter comes from whole-window RFM, which
includes the September period used to define the lapse label; the model's own
features remain limited to pre-cutoff data.

### Limitations and Next Steps

The lapse model and campaign metrics are based on one fixed train/test split
and historical synthetic order data; monitor performance and recalibrate
thresholds before a live campaign. The Premium flag is a post-hoc whole-window
label that includes the September lapse-label window, so the retention list is
illustrative and not a leakage-free evaluation; it must not be used to claim
prospective campaign performance. Evaluate future campaign outcomes with
time-based validation and an independently defined, pre-cutoff premium cohort.

## PART B ENDS ##


## PART C BEGINS ##

### Reproduce Part C

After Part B has produced `outputs/customer_segments.csv`, run:

```powershell
python -m src.triage
```

This processes `data/raw/tickets.csv` in file order using the supplied offline
`mock_llm.py`. It runs the zero-shot and few-shot comparisons, creates the
6000-token zero-shot budget demo, and writes the final 100000-token few-shot
output to `outputs/triage_results.csv`. The budget demo is written to
`outputs/triage_budget_demo.csv`.

### Input safety, redaction, and validation

All 164 ticket texts are redacted before eligibility checks; substitutions are
applied in order: email to `[EMAIL]`, phone to `[PHONE]`, then order ID to
`[ORDER_ID]`. The run found 55 email, 91 phone, and 132 order-ID substitutions
(278 total). It reapplies all three expressions to every redacted text and
asserts that there are zero remaining matches. Before each model call, the
code also asserts that the user message is exactly the redacted ticket text and
that neither the system nor user message contains any of the three patterns.
Empty/whitespace-only tickets are marked `skipped_empty`, and tickets whose
estimated input exceeds 400 tokens are marked `flagged_long`; neither is sent
to the model. The data contains four of each, and each long ticket estimates
to 734 tokens.

Both system prompts contain `CLASSIFY`, the four allowed categories
(`billing`, `delivery`, `technical`, `fraud`), and a JSON-only response shape.
The zero-shot prompt has no `EXAMPLES:` section. The few-shot prompt includes
three labeled examples, including a fraud example. Invalid JSON is accepted
only when regex-extracted lowercase words contain exactly one distinct allowed
category; valid JSON with an unsupported category and all other invalid
answers are flagged rather than guessed.

Every model call is retried up to three times on `TransientLLMError`, with
exponential backoff and jitter (`0.05 * 2**n * random.uniform(0.9, 1.15)`)
and one ticket/retry/wait log line. The mock produces 19 transient errors per
full run, all recovered on retry 0; waits are between 0.045 and 0.0575 seconds.
`mock_llm.reset()` runs at the start of each pipeline pass.

| Full run status | Zero-shot | Few-shot |
|---|---:|---:|
| `classified_json` | 135 | 135 |
| `classified_keyword` | 12 | 12 |
| `flagged_invalid_output` | 9 | 9 |
| `skipped_empty` | 4 | 4 |
| `flagged_long` | 4 | 4 |

### Prompt comparison

Accuracy is measured against `true_category` for classified tickets only;
token totals sum `response.usage.total_tokens`, including prompt and completion
tokens.

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 126 | 85.7% | 9,556 | 65.01 |
| Few-shot | 147 | 147 | 100.0% | 17,044 | 115.95 |

Zero-shot confusion table (classified tickets; rows are true categories):

| True \ Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

The zero-shot run classified 147 tickets correctly 126 times (85.7%) at 9,556
tokens, or 65.01 tokens per classified ticket. The few-shot run classified the
same 147 tickets correctly 147 times (100.0%) at 17,044 tokens, or 115.95
tokens per classified ticket. That is 50.94 extra tokens per classified ticket
and 7,488 extra tokens overall to correct all 21 errors, including 11 fraud
and 10 technical tickets otherwise routed as billing, which is worthwhile
when routing accuracy and fraud handling outweigh the additional prompt cost.

### Budget, routing, and human review

The zero-shot budget demo uses a 6,000-token budget and produces 58
`budget_exhausted` tickets. The budget is checked before each call, so the
last completed response can take the running total slightly past the threshold
(6,017 tokens here); no later tickets are sent after the threshold is reached.

The final `outputs/triage_results.csv` is the 100,000-token few-shot run and
has one row for each of the 164 tickets with columns `ticket_id`,
`customer_id`, `status`, `category`, `segment_name`, `priority`, `route`, and
`tokens_used`. Results with `classified_json` or `classified_keyword` have
passed the specified category validation and are routed to `security_team`
for fraud, `premium_desk` for other Premium customers, or `support_queue`
otherwise. Empty, long, invalid-output, and budget-exhausted tickets are not
classified for routing: they go to `human_review` at NORMAL priority. A
customer missing from the Part B segment file is treated as non-Premium.
Therefore the outputs explicitly distinguish rule-validated classifications
from tickets requiring human review; no invalid answer is silently guessed.

## PART C ENDS ##
