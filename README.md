# AOV recommender lab

Which product recommender raises the **order value** and the **real revenue** of an online shop?
Seven models are trained once, saved, and compared in a one-page Streamlit app. All money is in **so'm**.

## Data (real, no synthetic data)

Kaggle: [Online Retail II](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)
(`online_retail_II.xlsx`, also on UCI). Real transactions of a UK-based online gift-ware shop,
01/12/2009 – 09/12/2011. Many customers are wholesalers.

| Sheet | Rows | Columns |
|---|---|---|
| Year 2009-2010 | 525 461 | 8 |
| Year 2010-2011 | 541 910 | 8 |

Columns: `Invoice` (starts with `C` = cancellation), `StockCode`, `Description`, `Quantity`,
`InvoiceDate`, `Price` (£ per unit), `Customer ID`, `Country`. (CSV copies use `InvoiceNo` /
`UnitPrice` / `CustomerID` - both spellings work.) The two sheets overlap in Dec 2010; duplicates
are removed automatically. Cancellations, non-product codes, zero prices and returns are dropped.

**Rows are not orders.** The ~1.07 million rows are *line items* (one product on one invoice).
They form only **~39 500 orders** (about 15 different products each); after the time split,
~25 300 orders are used to train and ~10 200 to test. All 4 426 products seen in training are
used by default (they cover 95% of purchased items).

## Quick start

```bash
pip install -r requirements.txt
# 1. download online_retail_II.xlsx from Kaggle and put it into data/raw/
python train.py                 # reads data/raw/*, merges, trains ONCE, saves to artifacts/
streamlit run app.py            # the app only loads the saved models (seconds)
python run_experiment.py        # same comparison from the command line -> results/
pytest                          # 21 tests (use a tiny fake file, not your data)
```

The first `train.py` run reads the Excel file (slow: about 7 s per 100 000 rows in my test, so
roughly a minute or more for the full file) and caches it as Parquet in `data/cache/`; later
runs start in seconds. CSV and Parquet files work too, and every file in `data/raw/` is merged.

Options: `--fx 16100` (so'm per £1 - **check the Central Bank rate**), `--since 2011-01-01`,
`--top-k 500` (only the K most popular products; default: ALL - with 500 the carts keep just 48%
of the purchased items and percentages look inflated), `--currency UZS` (your own data already in so'm),
`--no-item-filter` (keep all item codes; the default keeps Online Retail's 5-digit product codes).

If something crashes oddly (e.g. `'Bundle' object has no attribute 'seq'`), run
`python3 check_install.py`: it detects old/mixed files, leftover `src/models/`, and fake data in
`data/raw/`. Always extract the zip into a **new** folder instead of copying files over an old one.

## The seven models

| Model | Idea |
|---|---|
| Random | Random products - the floor every model must beat. |
| Fixed list | The same best sellers for everyone (editable in the app). |
| Association rules | Products often bought together with the cart items. |
| Sequential | Next product from the customer's past orders and the cart (Markov chain). |
| Contextual bandit, cold start | LinUCB: learns what shoppers accept while running, from zero. |
| Contextual bandit, warm start | Same, pre-trained on past orders first. |

Every model returns at most 10 items: cart items removed, items above the price limit removed,
**shown cheapest first**.

## The simulation (shopper behaviour is assumed)

The data has no accept/reject logs, so shoppers are simulated. Half of a real test order is the
cart; the model suggests; the shopper accepts (wanted items more often, less often when pricey,
less often in lower positions), may remove items again, and then finishes or stops: **80% finish
when nothing is accepted; after accepting, about 30% stop**, more often when the add-on is
expensive relative to the cart. The app shows, per model: average order with suggestions, the
amount and % added, **real revenue change** (counts shoppers who left), acceptance, completion,
value per suggestion, and conversion by price band. Time-ordered split: the models learn from the
older 70% of orders and are tested on the newer 30%.

## Layout

```
check_install.py      detects old / mixed project files
train.py              train once -> artifacts/
app.py                Streamlit app (compare models + try a cart)
run_experiment.py     the same comparison from the command line
src/data.py           load Excel/CSV/Parquet, merge, clean, baskets, time split, so'm prices
src/models.py         the seven models
src/simulator.py      suggest -> accept -> remove -> finish or stop
src/artifacts.py      save / load (csv + npz + json, no pickle)
src/training.py       fit + save
tests/                21 tests (tests/fake_data.py is a tiny fixture, never used for training)
```

## Limits to keep in mind

* Accepting, removing and stopping are **assumptions**; results compare models under them and
  are not real-world gains. Real checkout / accept-reject logs would replace them.
* Prices are 2009–2011 UK prices converted at a fixed rate: a relabel, not an inflation or
  purchasing-power adjustment. Orders are wholesale-sized (about 15 products), unlike a typical
  consumer basket.
* The cart counts 1 unit per product, so "average order" is not the real invoice value.
