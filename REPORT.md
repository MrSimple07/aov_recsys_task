# Increasing Average Order Value (AOV) in App Sales
### A comparative study of Association Rules, Sequential Models and Contextual Bandits (simulation)

---

## 1. Goal

When a shopper is about to check out, the app can suggest **one extra item** ("Add this too?").
If the shopper accepts, the order becomes bigger. **AOV (average order value)** is the average
amount of money spent per order, and the aim of this project is to find which recommendation
method raises it the most.

I built **three ML models** and compared them in a **simulation**:

1. Association rules
2. Sequential model
3. Contextual bandit

I also added two simple baselines (Random and Popularity) so there is something to beat.

---

## 2. Dataset

I took the **Online Retail II** dataset from Kaggle (originally from the UCI Machine Learning
Repository). I used the 2009-2010 part, saved as `data/raw/online_retail_09_10.csv`.

| Column | Meaning | How I use it |
|---|---|---|
| `InvoiceNo` | Order ID | One invoice = one basket |
| `StockCode` | Product ID | Item identity |
| `Description` | Product name | Display in the UI |
| `Quantity` | Units bought | Cleaning (must be > 0) |
| `InvoiceDate` | Order time | Chronological split, time features |
| `UnitPrice` | Price per unit | Item price, AOV |
| `CustomerID` | Customer | Purchase history for the sequential model |
| `Country` | Country | Not used |

**Why this dataset:** it has baskets with several items, timestamps, customer IDs and **prices**.
Prices are required to measure AOV. I rejected Instacart (no prices) and Olist (almost every
order has a single item).

**Limitation:** it is data of a wholesale gift retailer, not of a mobile app. I state this
openly in the conclusions.

---

## 3. What I did, step by step

### Step 1: Data cleaning (`src/data.py`)
- Standardised column names (so `InvoiceNo`/`Invoice`, `UnitPrice`/`Price`, `CustomerID`/`Customer ID` all work).
- Removed rows with missing values.
- Removed **cancelled orders** (invoice numbers starting with `C`).
- Removed rows with quantity <= 0 or price <= 0.
- Removed non-product codes (postage, bank charges, manual adjustments), keeping only
  stock codes that start with 5 digits.
- Orders without a customer ID are treated as **guests** (no purchase history).

### Step 2: Baskets, split and vocabulary
- Each invoice became a **basket**: the list of unique items in the order they appear.
- I split the orders **by time**: the first **70%** of invoices are the **training period**
  (the past), the last **30%** are the **test period** (the future). No random shuffling, to
  avoid using the future to predict the past.
- I kept only the **top 500 most frequent items** of the training period (the vocabulary).
- The price of an item is its **median unit price** in the training period. One unit per item is assumed.
- Training baskets need at least 2 items, test baskets at least 3.

### Step 3: Model 1, Association Rules (`src/models/assoc_rules.py`)
- Ran **FP-Growth** on the training baskets (min support 0.005, itemsets up to 3 items).
- Turned the frequent itemsets into rules like *"A and B -> C"* with **confidence >= 0.1**
  and **lift > 1**.
- At recommendation time: find all rules whose left side is inside the cart, take the best
  confidence for each candidate item, keep the top-20 shortlist, and choose the one with the
  highest **score x price** (revenue-aware).
- If no rule matches, fall back to the best-selling item not in the cart.

### Step 4: Model 2, Sequential model (`src/models/sequential.py`)
- Built each customer's **chronological stream of purchased items**.
- Counted transitions "item i is followed by item j within the next 3 items", with decay 0.7 per step.
- Normalised into a **Markov chain** (transition probabilities).
- At recommendation time: take the customer's last 8 historical items plus the current cart
  (last 10 items total, recent ones weighted more, decay 0.8), sum their transition scores,
  then choose by **score x price**. Same popularity fallback.
- This is a simple, explainable sequential model. A neural one (GRU4Rec, SASRec) is a possible extension.

### Step 5: Model 3, Contextual bandit (`src/models/bandit.py`, `src/features.py`)
- Algorithm: **LinUCB** (one linear model per arm, optimistic exploration, alpha = 0.1).
- **Arms:** the 100 most popular items.
- **Context** (23 numbers): bias, cart size, cart value, weekday (sin/cos), hour (sin/cos),
  and the average **SVD embedding** (16 dimensions) of the items in the cart.
- **Reward:** accepted x price (scaled by the 95th price percentile), so it learns to maximise **money**, not just clicks.
- I evaluate two versions: **cold start** (knows nothing, learns during the test stream) and
  **warm start** (pre-trained inside the simulator on past orders).

### Step 6: Baselines
- **Random:** a random item not in the cart.
- **Popularity:** the best-selling item not in the cart.

### Step 7: Simulator (`src/simulator.py`)
The dataset has only finished orders, so it cannot tell whether a shopper *would accept* a
suggestion. So I **simulate** the shopper:

1. Take a real order from the **test period**.
2. Randomly put **50% of its items into the cart**. The other 50% are the shopper's **latent wants**.
3. The model sees only the cart (plus history, time) and suggests an item.
4. The shopper accepts with probability:

   ```
   p = (p_hit if the item is a latent want else p_miss) * exp(-s * price / median_price)
   ```
   with `p_hit = 0.5`, `p_miss = 0.03`, price sensitivity `s = 0.15`.
5. If accepted, the item's price is added to the order.
6. Repeat for up to 10,000 test orders and **5 random seeds** (different cart splits and random draws).

**Fairness:** every policy is evaluated on the **same sessions and the same random numbers**
(common random numbers), and the bandit keeps learning online during the test stream.

### Step 8: Metrics
- **AOV without recommendations** (the cart value only) and **AOV with recommendations**
- **Uplift** in money and in %, the main metric
- **Acceptance rate:** share of suggestions accepted
- **Relevance rate:** share of suggestions that were latent wants
- **95% confidence interval** of the uplift over the seeds

### Step 9: Sensitivity analysis
Because the shopper behaviour is **assumed**, I re-ran the comparison while changing one assumption at a time:
`p_hit` in {0.3, 0.5, 0.7}, `p_miss` in {0.01, 0.03, 0.08}, price sensitivity in {0, 0.15, 0.4}.
The question is whether the **ranking of the models** stays the same.

### Step 10: Testing
6 automated tests (`pytest`) check cleaning, the chronological split, the simulator (an
oracle policy must beat random), the bandit (it must learn the best arm), determinism of the
experiment, and that the Streamlit app runs without errors.

---

## 4. Results

> Fill this section in with **your own numbers** after running on the real dataset
> (`results/summary.csv`, `results/aov_uplift.png`, `results/learning_curves.png`,
> `results/sensitivity.png`).

| Policy | AOV without recs | AOV with recs | Uplift (%) | Acceptance rate | Relevance rate |
|---|---|---|---|---|---|
| Random | | | | | |
| Popularity | | | | | |
| Association Rules | | | | | |
| Sequential (Markov) | | | | | |
| LinUCB (cold start) | | | | | |
| LinUCB (warm start) | | | | | |

**How to read it**
- Every real model should beat Random and Popularity. If one does not, it is a finding to discuss.
- The **cold-start bandit** usually starts weak and improves over time (see the learning curve),
  because it has to learn by trial and error. The warm-start version shows the benefit of pre-training.
- Check `sensitivity.png`: if the ranking of the models stays the same when the assumptions change,
  the conclusion is robust.

---

## 5. Limitations (state these in the conclusions)

- Shopper behaviour is **simulated**. The absolute uplift depends on `p_hit`, `p_miss` and price
  sensitivity, so the **ranking** of the methods matters more than the exact percentages.
- "Latent wants" are the rest of the real basket, which favours methods that capture co-purchases.
- The dataset is a wholesale retailer, not an app.
- The bandit can choose only among the top-100 popular items, while the other two models can choose from all 500.
- Item order **inside** an invoice follows the file order and may not be a real time order. The order **across** invoices is real.
- A customer's history is fixed to the training period. Guests have no history.
- One unit per item and median prices are assumed.

**Possible extensions:** Thompson sampling, a neural sequential model (GRU4Rec/SASRec), several
recommendation slots, and off-policy evaluation on real logged A/B data.

---

## 6. Project structure

```
aov-recsys/
├── app.py                    # Streamlit UI (explanation, live demo, simulation)
├── run_experiment.py         # Same experiment from the command line -> results/
├── requirements.txt          # Python dependencies
├── README.md                 # Short project description
├── REPORT.md                 # This document
├── data/
│   └── raw/                  # Put the Kaggle CSV here (any file name)
├── results/                  # Output of run_experiment.py (tables and charts)
├── src/
│   ├── config.py             # SimConfig: simulation assumptions
│   ├── data.py               # Loading, cleaning, baskets, time split, vocabulary
│   ├── features.py           # SVD item embeddings + bandit context builder
│   ├── simulator.py          # Shopper simulator and metrics
│   ├── experiment.py         # Trains the models once, evaluates all policies
│   ├── plots.py              # Matplotlib charts for results/
│   ├── synthetic.py          # Fake data generator (tests / smoke test only)
│   └── models/
│       ├── base.py           # Session, Policy interface, Random & Popularity baselines
│       ├── assoc_rules.py    # Model 1: FP-Growth association rules
│       ├── sequential.py     # Model 2: decayed Markov chain
│       └── bandit.py         # Model 3: LinUCB contextual bandit
└── tests/
    ├── test_pipeline.py      # Cleaning, split, simulator, bandit, determinism
    └── test_app.py           # Runs the Streamlit app headlessly
```

---

## 7. How to run

### Install
```bash
cd aov-recsys
pip install -r requirements.txt
```
Put the Kaggle CSV into `data/raw/`. Any file name works, the first `.csv` or `.xlsx` is used.

> Messages like `'parseString' deprecated` from matplotlib are harmless warnings from the local matplotlib installation.

### Option A: Streamlit UI (recommended)
```bash
streamlit run app.py
```
It opens at `http://localhost:8501`. The first start takes about a minute because it trains the three models. After that it is cached.

### Option B: Command line
```bash
python run_experiment.py                    # main comparison
python run_experiment.py --sensitivity      # plus sensitivity analysis
```
Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--data` | first file in `data/raw` | Path to the dataset |
| `--top-k` | 500 | Number of most popular items kept |
| `--seeds` | 5 | Number of simulation repeats |
| `--max-test` | 10000 | Max test orders per repeat |
| `--min-support` | 0.005 | Minimum support of association rules |
| `--min-confidence` | 0.1 | Minimum confidence of rules |
| `--arms` | 100 | Number of bandit arms |
| `--alpha` | 0.1 | LinUCB exploration strength |
| `--slots` | 1 | Suggestions shown per order |
| `--p-hit` | 0.5 | Accept probability for a wanted item |
| `--p-miss` | 0.03 | Accept probability for an unrelated item |
| `--price-sensitivity` | 0.15 | How strongly price lowers acceptance |

Outputs in `results/`: `summary.csv`, `summary.md`, `per_seed.csv`, `aov_uplift.png`,
`learning_curves.png`, and with `--sensitivity` also `sensitivity.csv` and `sensitivity.png`.

### Tests
```bash
pytest
```

---

## 8. The Streamlit UI: every control explained

### Sidebar (always visible)
| Control | What it does |
|---|---|
| **Dataset file** | Chooses which file in `data/raw/` to use. |
| **Products to keep (most popular)** | Slider 100-1000. How many top items form the vocabulary. Fewer = faster. |
| **Rule min. support** | A rule must appear in at least this share of orders. Lower = more rules but slower. |

Changing any sidebar setting re-trains the models (the first run is cached afterwards).

### Tab 1: How it works
Read-only explanation:
- The idea (cart, suggestion, AOV) and the three methods in three columns.
- Why a simulation is needed, and the 5 simulation steps.
- **Your data** panel: *Products used*, *Training orders (past)*, *Test orders (future)*, *Known customers*, and the number of association rules learned.
- How to read the results (uplift, acceptance rate, relevant picks, baselines, the warning about assumed behaviour).

### Tab 2: Try it live
Build a cart yourself and see what each model suggests.

| Control | What it does |
|---|---|
| **Items in the cart** | Searchable multi-select (max 15 items), sorted by popularity, shown as `Name [code] · £price`. |
| **Customer** | `Guest (no history)` or a known customer ID. A known customer gives the sequential model their purchase history. |
| **Weekday** / **Hour** | The time context, used by the bandit. |
| **Suggestions per model** | 1 to 5 items shown per model. |
| **Current cart value** | Total price of the selected items. |

Below, four columns show the suggestions of **① Association rules**, **② Sequential (Markov)**,
**③ Contextual bandit** and the **Baseline: best sellers**, each as a small table with *Item*,
*Price* and (for rules and Markov) a *Score*. The bandit has no score because it picks by learned expected revenue.
The bandit used here is pre-trained in the simulator.

### Tab 3: Run the simulation
**Assumption sliders (shopper behaviour)**

| Control | Meaning |
|---|---|
| **p_hit** (0.1-0.9) | Chance of accepting an item the shopper wanted. |
| **p_miss** (0-0.2) | Chance of accepting an unrelated item (impulse buy). |
| **Price sensitivity** (0-1) | Higher = expensive items are accepted less often. |

**Run settings**

| Control | Meaning |
|---|---|
| **Suggestions shown** (1-3) | Slots per order. |
| **Repeats (random seeds)** (1-8) | More repeats = more reliable but slower. |
| **Orders per repeat** | 1,000 to 10,000 test orders. |
| **Bandit exploration (alpha)** | How much the bandit explores. |

**▶ Run simulation** button runs everything and shows:
- A green box with the **best policy** and its AOV before and after.
- A **results table** (AOV with and without recommendations, added money per order, uplift %, acceptance rate, relevant picks, 95% CI).
- A **bar chart** of AOV uplift (%) per policy.
- A **line chart** of performance over time, showing how the bandit learns.
- A **Download results (CSV)** button.

The results stay on screen until you run the simulation again.

---

## 9. Conclusion (template)

> Using the Online Retail II dataset, I compared association rules, a sequential Markov model and a
> LinUCB contextual bandit in a shopper simulation. **[Best method]** achieved the highest AOV uplift
> (**[X]%**), followed by **[...]**. The cold-start bandit needed time to learn but **[improved / stayed behind]**,
> while the pre-trained bandit **[...]**. The ranking **[stayed / changed]** under the sensitivity analysis.
> Since the shopper behaviour is simulated, the results show the relative strength of the methods
> rather than guaranteed real-world gains, and the next step would be an A/B test in a real app.