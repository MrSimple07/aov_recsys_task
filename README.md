# Increasing AOV in app sales: Association Rules vs Sequential Models vs Contextual Bandits

Simulation study comparing three recommenders on **average order value (AOV)**.

## Dataset: Online Retail II (UCI / Kaggle)
Search Kaggle for **"Online Retail II UCI"** (or the UCI ML Repository) and save the file as
`data/raw/` (any file name, `.csv` or `.xlsx`). Why this dataset: invoices (baskets),
timestamps, customer IDs (purchase sequences) and **unit prices** (needed for AOV).
Caveat: it is a wholesale/gift retailer, not a mobile app. State this in the report.
`python -m src.synthetic` (writes to data/raw/) creates a fake dataset in the same format for a quick smoke test.
**Results on synthetic data are only a pipeline demo.**

## Quick start
```bash
pip install -r requirements.txt
# put the Kaggle CSV into data/raw/ (any file name)
streamlit run app.py                       # interactive UI: explanation, live demo, simulation
python run_experiment.py --sensitivity     # same experiment from the command line -> results/
pytest
```
Outputs go to `results/`: `summary.csv/.md`, `per_seed.csv`, `aov_uplift.png`,
`learning_curves.png`, `sensitivity.csv/.png`.
Useful flags: `--top-k 500 --seeds 5 --slots 1 --alpha 0.1 --min-support 0.005 --arms 100`.

## Pipeline
1. **Clean** (`src/data.py`): drop cancellations, non-positive qty/price, non-product codes.
2. **Split chronologically** by invoice (70% train / 30% test). Vocabulary = top-K train items.
   Item price = median unit price (1 unit per item).
3. **Models** (`src/models/`), all choosing 1 item for the cart (`--slots` for more):
   - *Association rules*: FP-Growth, rules with 1-2 item antecedents, score = confidence, filtered by lift > 1.
   - *Sequential*: decayed Markov chain over each customer's chronological item stream
     (+ current cart as the most recent items). Upgrade path: GRU4Rec/SASRec.
   - *Contextual bandit*: disjoint LinUCB, arm = one of the top-100 popular items, context =
     cart size, cart value, weekday/hour, mean SVD embedding of cart items.
     Reward = accepted x price (scaled). Reported cold-start and warm-start (pre-trained in the simulator on train sessions).
   - Baselines: Random, Popularity (best seller not in cart).
   - Association rules and Markov rank their shortlist by score x price (revenue-aware);
     both fall back to popularity if nothing matches.
4. **Simulator** (`src/simulator.py`): each test basket = one checkout session. Random half is
   the *cart*, the rest are *latent wants*. Shown item accepted with
   `p_hit` (latent want) or `p_miss` (not), multiplied by `exp(-s * price/median_price)`.
   Accepted items add their price. Uniform random draws are shared across policies (common random
   numbers), and everything is repeated over seeds (different cart splits and draws).
5. **Metrics**: AOV, AOV uplift (abs and %), acceptance rate, relevance rate, 95% CI over seeds.

## Assumptions and limitations (write these in the report)
- Acceptance behaviour is **simulated**, not observed. Absolute uplift depends on `p_hit`, `p_miss`,
  `price_sensitivity`; use `--sensitivity` and compare *rankings*, not absolute numbers.
- "Latent wants" = the rest of the real basket, which favours models that capture co-purchase.
- Bandit arms are limited to top-N items; AR/Markov may recommend any vocabulary item.
- Row order inside an invoice is used as order for the sequential model; cross-invoice order is real.
- Customer history is static (train period only); guests have no history.
- Next steps: Thompson sampling, neural sequential model, multiple slots, off-policy evaluation.

## Report outline
Intro and AOV motivation -> Data -> Methods (3 models) -> Simulation design and assumptions ->
Results (table + figures) -> Sensitivity -> Limitations -> Conclusion.
