"""Streamlit app:  streamlit run app.py"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import SimConfig
from src.data import load_dataset
from src.experiment import _subsample, aggregate, evaluate, prepare
from src.models import LinUCBPolicy, PopularityPolicy, Session
from src.simulator import build_sessions, run_policy

st.set_page_config(page_title="AOV Recommender Lab", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------- sidebar
def list_files():
    env = os.environ.get("AOV_DATA")
    if env:
        return [env]
    folder = Path("data/raw")
    return sorted(str(p) for p in list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")))


files = list_files()
st.sidebar.title("Settings")
if not files:
    st.error("No dataset found. Put the Kaggle CSV into `data/raw/` and refresh.")
    st.stop()
path = st.sidebar.selectbox("Dataset file", files)
top_k = st.sidebar.slider("Products to keep (most popular)", 100, 1000, 500, 50,
                          help="Rare products are dropped; fewer products = faster.")
min_support = st.sidebar.select_slider(
    "Rule min. support", [0.002, 0.003, 0.005, 0.01, 0.02], value=0.005,
    help="A rule must appear in at least this share of orders. Lower = more rules, slower.")


@st.cache_resource(show_spinner="Loading data and training the models (can take a minute)...")
def load_and_train(path, top_k, min_support):
    return prepare(load_dataset(path, top_k), min_support=min_support)


@st.cache_resource(show_spinner="Pre-training the bandit inside the simulator...")
def warm_bandit(_prep, key, alpha):
    ds, cfg = _prep.ds, SimConfig()
    pol = LinUCBPolicy(_prep.arms, ds.prices, _prep.ctxb.dim, alpha, name="LinUCB (bandit)")
    warm = build_sessions(_subsample(ds.train, 15000), _prep.ctxb, ds.prices, cfg, np.random.default_rng(0))
    run_policy(pol, warm, ds.prices, cfg, learn=True)
    return pol


prep = load_and_train(path, top_k, min_support)
ds = prep.ds

st.title("🛒 Raising Average Order Value with Recommendations")
tab_how, tab_live, tab_sim = st.tabs(["1 · How it works", "2 · Try it live", "3 · Run the simulation"])

# ---------------------------------------------------------------- tab 1
with tab_how:
    st.header("The idea in one minute")
    st.markdown("""
A shopper has some items in the **cart**. Just before checkout the app shows **one extra
item** ("Add this too?"). If the shopper accepts, the order gets bigger.
**AOV (average order value)** = the average money spent per order. The goal is to pick the
suggestion that raises it the most.

We compare **three ways of choosing that suggestion**:
""")
    c1, c2, c3 = st.columns(3)
    c1.subheader("① Association rules")
    c1.markdown("Looks at past orders and finds items **bought together** "
                "(*\"people with A and B also buy C\"*). Simple, explainable, no learning at runtime.")
    c2.subheader("② Sequential model")
    c2.markdown("Looks at **what a customer bought, in order**, and predicts what usually "
                "comes next (a Markov chain). Uses the customer's history plus the current cart.")
    c3.subheader("③ Contextual bandit")
    c3.markdown("A model that **learns by trial and error**. It tries suggestions, sees whether "
                "they were accepted (and how much money they brought), and adapts. It also uses "
                "context: cart size, cart value, weekday, hour, type of items.")

    st.header("Why a simulation?")
    st.markdown("""
Our dataset only contains **finished orders**. It never recorded *\"we suggested X, did the shopper
accept?\"*, so we can't measure uplift directly. So we **simulate shoppers**:

1. Take a real finished order from the **test period** (orders the models never saw).
2. Randomly put **half of its items into the cart**. The other half are the shopper's **hidden wants**
   (things they would plausibly like).
3. Each model sees only the cart (and history) and suggests one item.
4. The simulated shopper **accepts** with probability `p_hit` if the item is one of their hidden wants,
   or a small `p_miss` if not (impulse buy). More expensive items are accepted a bit less often.
5. If accepted, the item's price is added to the order. Repeat for thousands of orders and several
   random seeds, then compare **AOV with vs. without recommendations**.

All models face the **same random numbers**, so the comparison is fair.
""")
    st.header("Data")
    n_cust = len(prep.seq.history)
    a, b, c, d = st.columns(4)
    a.metric("Products used", ds.V)
    b.metric("Training orders (past)", f"{len(ds.train):,}")
    c.metric("Test orders (future)", f"{len(ds.test):,}")
    d.metric("Known customers", f"{n_cust:,}")
    st.caption(f"Association rules learned: **{prep.ar.n_rules:,}**. Orders are split by time: "
               "first 70% to learn from, last 30% to test on, like in real life.")
    st.header("How to read the results")
    st.markdown("""
* **Uplift (%)**: how much higher AOV is with the recommender than without. Higher is better.
* **Acceptance rate**: share of suggestions the simulated shopper accepted.
* **Relevant picks**: share of suggestions that were one of the shopper's hidden wants.
* **Random** and **Popularity** are baselines. A real model should beat them.
* ⚠️ The shopper's behaviour is **assumed**, not observed. Trust the **ranking** of the models and
  check it still holds when you change the assumptions (tab 3), more than the exact percentages.
""")

# ---------------------------------------------------------------- tab 2
with tab_live:
    st.header("Build a cart and see what each model suggests")
    order = list(prep.pop_order)

    def label(i):
        name = (ds.names[i] or ds.item_ids[i]).strip().title()
        return f"{name} [{ds.item_ids[i]}] · £{ds.prices[i]:.2f}"

    labels = {label(i): int(i) for i in order}
    left, right = st.columns([3, 1])
    chosen = left.multiselect("Items in the cart (type to search)", list(labels), max_selections=15)
    customers = ["Guest (no history)"] + [str(c) for c in sorted(prep.seq.history)[:300]]
    cust = right.selectbox("Customer", customers,
                           help="Known customers give the sequential model their purchase history.")
    c1, c2, c3 = st.columns(3)
    weekday = c1.selectbox("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=2)
    hour = c2.slider("Hour", 6, 22, 12)
    n_show = c3.slider("Suggestions per model", 1, 5, 3)

    if not chosen:
        st.info("Add at least one item to the cart.")
    else:
        cart = np.array([labels[x] for x in chosen])
        ts = pd.Timestamp("2010-06-14") + pd.Timedelta(days=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(weekday), hours=hour)
        customer = -1 if cust.startswith("Guest") else int(cust)
        sess = Session(cart, customer, ts, prep.ctxb.build(cart, ts))
        st.metric("Current cart value", f"£{ds.prices[cart].sum():.2f}")

        bandit = warm_bandit(prep, (path, top_k, min_support), 0.1)
        models = [(prep.ar, "① Association rules"), (prep.seq, "② Sequential (Markov)"),
                  (bandit, "③ Contextual bandit"), (PopularityPolicy(prep.pop_order), "Baseline: best sellers")]
        cols = st.columns(4)
        for col, (pol, title) in zip(cols, models):
            col.subheader(title)
            recs = pol.recommend(sess, n_show)
            rows = []
            for i in recs:
                row = {"Item": (ds.names[i] or ds.item_ids[i]).strip().title(), "Price": f"£{ds.prices[i]:.2f}"}
                if hasattr(pol, "scores"):
                    row["Score"] = round(float(pol.scores(sess)[i]), 3)
                rows.append(row)
            col.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Association rules / Markov rank their shortlist by score × price, so pricier items win "
                   "among similarly likely ones. Score ≈ how strongly the model links the cart to the item "
                   "(for rules: confidence). The bandit has no score because it picks by learned expected revenue.")

# ---------------------------------------------------------------- tab 3
with tab_sim:
    st.header("Simulate shoppers and compare AOV uplift")
    st.markdown("These sliders are the **assumptions** about shopper behaviour.")
    a, b, c = st.columns(3)
    p_hit = a.slider("p_hit: accept an item they wanted", 0.1, 0.9, 0.5, 0.05)
    p_miss = b.slider("p_miss: accept an unrelated item", 0.0, 0.2, 0.03, 0.01)
    price_s = c.slider("Price sensitivity", 0.0, 1.0, 0.15, 0.05,
                       help="Higher = expensive items are accepted less often.")
    d, e, f, g = st.columns(4)
    slots = d.slider("Suggestions shown", 1, 3, 1)
    seeds = e.slider("Repeats (random seeds)", 1, 8, 3)
    max_test = f.select_slider("Orders per repeat", [1000, 2000, 3000, 5000, 10000], value=3000)
    alpha = g.select_slider("Bandit exploration (alpha)", [0.02, 0.05, 0.1, 0.2, 0.3], value=0.1)

    if st.button("▶ Run simulation", type="primary"):
        cfg = SimConfig(p_hit=p_hit, p_miss=p_miss, price_sensitivity=price_s, slots=slots)
        with st.spinner("Simulating shoppers..."):
            df, curves = evaluate(prep, cfg, seeds, alpha, max_test)
        st.session_state["result"] = (aggregate(df), curves)

    if "result" in st.session_state:
        summary, curves = st.session_state["result"]
        best = summary.loc[summary["uplift_pct"].idxmax()]
        st.success(f"Best policy: **{best['policy']}** with **+{best['uplift_pct']:.2f}%** AOV "
                   f"(from £{best['AOV_base']:.2f} to £{best['AOV_policy']:.2f} per order).")
        show = summary.rename(columns={
            "policy": "Policy", "AOV_base": "AOV no recs (£)", "AOV_policy": "AOV with recs (£)",
            "uplift_abs": "Added per order (£)", "uplift_pct": "Uplift (%)",
            "accept_rate": "Acceptance rate", "relevance_rate": "Relevant picks",
            "uplift_pct_ci95": "± 95% CI (pp)"})
        st.dataframe(show.round(3), hide_index=True, use_container_width=True)
        st.subheader("AOV uplift (%)")
        st.bar_chart(summary.set_index("policy")["uplift_pct"])
        st.subheader("Performance over time (rolling average of £ added per order)")
        win = max(50, min(500, len(next(iter(curves.values()))[0]) // 5))
        curve_df = pd.DataFrame({k: pd.Series(np.mean(v, axis=0)).rolling(win).mean() for k, v in curves.items()})
        st.line_chart(curve_df.dropna())
        st.caption("The cold-start bandit starts with no knowledge and improves as orders arrive; "
                   "the warm-start one was pre-trained inside the simulator on past orders.")
        st.download_button("Download results (CSV)", summary.to_csv(index=False), "summary.csv")
