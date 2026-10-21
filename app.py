"""AOV recommender lab - which model raises the order value?   Run:  streamlit run app.py"""
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.artifacts import load_artifacts
from src.config import SimConfig, som
from src.experiment import make_policies, run_simulation
from src.models import Ctx, relevance

st.set_page_config(page_title="Which recommender earns more?", page_icon="🛒", layout="wide")
ART = os.environ.get("AOV_ARTIFACTS", "artifacts")

if not (Path(ART) / "meta.json").exists():
    st.error("No trained models found. Train once, then reload this page:")
    st.code("python train.py --data data/raw", language="bash")
    st.stop()


@st.cache_resource
def get_bundle(path):
    return load_artifacts(path)


@st.cache_data(show_spinner=False)
def simulate(path, cfg_items, orders, repeats, fixed_items):
    return run_simulation(get_bundle(path), SimConfig(**dict(cfg_items)), repeats, orders,
                          list(fixed_items))


@st.cache_data(show_spinner=False)
def stop_slope(path, cfg_items):
    return estimate_stop_slope(get_bundle(path), SimConfig(**dict(cfg_items)))


def pct(v):
    return f"{v:+.1f}%"


b = get_bundle(ART)
cat, rules, meta = b.cat, b.rules, b.meta

st.title("🛒 Which recommender earns the most?")
st.caption("We replay real orders from the test period, let each model suggest products at "
           "checkout, and simulate whether shoppers accept them. All money is in so'm.")
st.info("**How to read this page**  ① Pick the situation below.  ② The table shows, for every model, "
        "how much it adds to the average order.  ③ **Real revenue** also counts shoppers who left "
        "because the price felt too high - this is what the business actually earns.")

# --------------------------------------------------------------------- step 1: the situation
st.subheader("① Choose the situation")
pmin, pmax = int(cat.prices.min()), int(cat.prices.max())
step = 1000 if pmax > 100_000 else 100
default_limit = int(min(max(round(np.quantile(cat.prices, 0.9), -3), pmin), pmax))
labels = {f"{cat.label(i)} · {som(cat.prices[i])}": int(i) for i in cat.pop_order[:300]}

with st.form("settings"):
    c1, c2, c3, c4 = st.columns(4)
    n_slots = c1.slider("Suggestions shown", 1, 10, 10)
    c1.caption("How many products appear at checkout (up to 10).")
    limit = c2.slider("Highest price allowed (so'm)", pmin, pmax, default_limit, step)
    no_limit = c2.checkbox("No price limit", value=False)
    c2.caption("Pricier products are never suggested. The list is shown cheapest first.")
    s = c3.select_slider("Price sensitivity", [0.0, 0.15, 0.4, 0.8], value=0.15,
                         format_func=lambda v: {0.0: "None", 0.15: "Low", 0.4: "Medium", 0.8: "High"}[v])
    c3.caption("How much a higher price lowers the chance that a shopper accepts.")
    stop = c4.select_slider("Shoppers who stop after accepting", [0.20, 0.25, 0.30, 0.40], value=0.30,
                            format_func=lambda v: f"{int(round(v * 100))}%")
    c4.caption("Share who leave because it feels too pricey. 20% = same as without suggestions.")

    with st.expander("More settings (optional)"):
        a1, a2, a3 = st.columns(3)
        finish = a1.slider("Orders that finish without suggestions", 0.5, 1.0, 0.80, 0.05)
        a1.caption("The rest leave without buying (default 20%).")
        p_hit = a2.slider("Accept a product they wanted", 0.1, 1.0, 0.50, 0.05)
        a2.caption("Chance to accept a suggestion the shopper was planning to buy.")
        p_miss = a3.slider("Accept an unrelated product", 0.0, 0.3, 0.03, 0.01)
        a3.caption("Chance to accept something unplanned (often removed again).")
        b1, b2, b3 = st.columns(3)
        orders = b1.select_slider("Orders replayed per repeat", [500, 1000, 1500, 3000, 5000], value=1500)
        b1.caption("More orders = steadier numbers but slower.")
        repeats = b2.select_slider("Repeats", [1, 2, 3, 5], value=2)
        b2.caption("Repeating shows how stable the result is (the ± in the table).")
        fixed_sel = b3.multiselect("Fixed list products", list(labels), default=list(labels)[:10],
                                   max_selections=20)
        b3.caption("Shown to everyone in the 'Fixed list' model. Default: 10 best sellers.")
    st.form_submit_button("Update results", type="primary")

if s > 0:
    st.caption(f"With this price sensitivity, the price that earns the most per suggestion is "
               f"about **{som(cat.ref / s)}**.")
cfg = SimConfig(n_slots=n_slots, price_limit=float("inf") if no_limit else float(limit),
                price_sensitivity=s, p_hit=p_hit, p_miss=p_miss, finish_base=finish,
                stop_after_accept=stop)
cfg_items = tuple(sorted(asdict(cfg).items()))
fixed_items = tuple(labels[x] for x in fixed_sel) or tuple(int(i) for i in cat.pop_order[:10])
pols = [p for p in make_policies(b, cfg, list(fixed_items)) if p.name != "Expected value"]
blurb = {p.name: p.blurb for p in pols}

tab_cmp, tab_cart = st.tabs(["📊 Compare models", "🧺 Try a cart"])

# ------------------------------------------------------------------------- step 2: the table
with tab_cmp:
    with st.spinner("Simulating all models (a few seconds)..."):
        summary, bands = simulate(ART, cfg_items, orders, repeats, fixed_items)
    summary = summary[summary["policy"] != "Expected value"].reset_index(drop=True)
    bands = bands[bands["policy"] != "Expected value"]
    base_aov = summary["AOV_base"].iloc[0]
    
    st.subheader("② What each model adds")
    best_aov = summary.loc[summary["AOV_uplift_pct"].idxmax()]
    best_rev = summary.loc[summary["RPS_uplift_pct"].idxmax()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Average order without suggestions", som(base_aov))
    m2.metric(f"Biggest order increase: {best_aov['policy']}",
              f"+{som(best_aov['extra_per_order'])}", pct(best_aov["AOV_uplift_pct"]))
    m3.metric(f"Best real revenue: {best_rev['policy']}",
              pct(best_rev["RPS_uplift_pct"]), f"± {best_rev['RPS_uplift_pct_ci95']:.1f}", delta_color="off")
    if best_rev["RPS_uplift_pct"] > best_rev["RPS_uplift_pct_ci95"] and best_rev["RPS_uplift_pct"] > 0.5:
        st.success(f"**{best_rev['policy']}** earns the most real revenue: "
                   f"{som(best_rev['extra_per_1000_started'])} more per 1 000 started orders.")
    else:
        st.warning("No model clearly increases real revenue in this situation: the larger orders are "
                   "cancelled out by shoppers who leave. Try a lower price limit, lower price "
                   "sensitivity, or fewer shoppers stopping.")

    tbl = pd.DataFrame({
        "Model": summary["policy"],
        "What it does": summary["policy"].map(blurb),
        "Average order with suggestions": summary["AOV_policy"],
        "Added to order (so'm)": summary["extra_per_order"],
        "Added to order (%)": summary["AOV_uplift_pct"],
        "Real revenue change (%)": summary["RPS_uplift_pct"],
        "± (%)": summary["RPS_uplift_pct_ci95"],
        "Suggestions accepted (%)": 100 * summary["conversion"],
        "Orders completed (%)": 100 * summary["finish_rate"],
        "Value per suggestion (so'm)": summary["value_per_suggestion"]})
    money = lambda v: f"{v:,.0f}".replace(",", " ")  # noqa: E731
    sty = (tbl.style.format({"Average order with suggestions": money, "Added to order (so'm)": money,
                             "Value per suggestion (so'm)": money, "Added to order (%)": "{:+.1f}%",
                             "Real revenue change (%)": "{:+.1f}%", "± (%)": "{:.1f}",
                             "Suggestions accepted (%)": "{:.1f}%", "Orders completed (%)": "{:.1f}%"})
           .highlight_max(subset=["Added to order (so'm)", "Added to order (%)", "Real revenue change (%)",
                                  "Value per suggestion (so'm)"],
                          props="background-color:#1b7f3b;color:#ffffff;font-weight:600"))
    st.dataframe(sty, hide_index=True, use_container_width=True)
    st.caption("**Added to order** = how much larger the average *finished* order gets. "
               "**Real revenue change** = money per *started* order, including shoppers who left - "
               "the true business result. **Value per suggestion** = money kept in carts ÷ suggestions shown. "
               "Green = best in column. Random shows how much is just noise.")
    st.download_button("Download table (CSV)", summary.to_csv(index=False), "model_comparison.csv")

    chart = tbl.set_index("Model")[["Added to order (%)", "Real revenue change (%)"]]
    st.bar_chart(chart, y_label="% change vs no suggestions")
    st.caption("The gap between the two bars is shoppers who stopped because of price.")

    # ------------------------------------------------------------- step 3: price vs conversion
    st.subheader("③ Price vs conversion")
    st.caption("Cheap products are accepted more often, but value = chance to accept × price. "
               "A product accepted 70% of the time at 5 earns 3.5; one accepted 30% at 30 earns 9.")
    names = list(summary["policy"])
    pick = st.selectbox("Model", names, index=names.index(best_rev["policy"]))
    bb = bands[bands["policy"] == pick].set_index("band")
    cc = st.columns(2)
    cc[0].markdown("**Suggestions accepted (%), by price band**")
    cc[0].bar_chart(100 * bb["conversion"])
    cc[1].markdown("**Value per suggestion (so'm), by price band**")
    cc[1].bar_chart(bb["value_per_suggestion"])
    if bb["shown"].iloc[0] > 0 and bb["shown"].iloc[3] > 0:
        lo, hi = bb.iloc[0], bb.iloc[3]
        st.caption(f"{pick}: the cheapest quarter of products is accepted {100 * lo['conversion']:.1f}% of the "
                   f"time and earns {som(lo['value_per_suggestion'])} per suggestion; the priciest quarter is "
                   f"accepted {100 * hi['conversion']:.1f}% and earns {som(hi['value_per_suggestion'])}.")

# ----------------------------------------------------------------------------- try a cart
with tab_cart:
    st.caption("Build a cart and see what a model would suggest. The best business value is always first.")
    names_live = [p.name for p in pols]
    model = st.selectbox("Model", names_live)
    opts = {f"{cat.label(i)} · {som(cat.prices[i])}": int(i) for i in cat.pop_order[:300]}
    picked = st.multiselect("Products in the cart", list(opts), max_selections=15, default=list(opts)[20:23])
    cart = np.array([opts[x] for x in picked], dtype=int)

    seq = next((p for p in pols if p.name == "Sequential"), None)
    hist = seq.model.history if seq is not None else {}
    who = st.radio("Shopper", ["New shopper (no history)", "Returning shopper"], horizontal=True)
    customer = -1
    if who.startswith("Returning"):
        if not hist:
            st.info("No customer histories were saved with the models.")
        else:
            ids = sorted(hist, key=lambda c: -len(hist[c]))[:200]
            customer = st.selectbox("Customer", ids,
                                    format_func=lambda c: f"Customer {c} · {len(hist[c])} past items")
            st.caption("Recently bought: " + ", ".join(cat.label(int(i)) for i in hist[customer][-8:]))
            if model != "Sequential":
                st.caption("ℹ Only the **Sequential** model uses purchase history; this model ignores it.")

    if len(cart) == 0:
        st.info("Add at least one product to the cart.")
    else:
        pol = next(p for p in pols if p.name == model)
        st.write(f"Cart value: **{som(cat.prices[cart].sum())}**")
        st.caption(blurb.get(pol.name, pol.blurb))
        if "cold" in pol.name.lower() and "bandit" in pol.name.lower():
            st.warning("Cold-start bandit has learned nothing yet here, so its picks are close to random. "
                       "It only improves during the simulation in the comparison table.")
        ts = time.time_ns()
        vec = pol.ctxb.build(cart, ts) if hasattr(pol, "ctxb") else None
        recs = pol.recommend(cart, n_slots, cfg.price_limit, Ctx(customer, ts, vec))
        if not recs:
            st.write("Nothing to suggest under this price limit.")
        else:
            rel = relevance(rules, cat, cart)
            chance = rel * cfg.p_hit * np.exp(-cfg.price_sensitivity * cat.prices / cat.ref)
            value = chance * cat.prices                       # business value = chance x price
            recs = sorted((int(i) for i in recs), key=lambda i: -value[i])   # best value first
            st.dataframe(pd.DataFrame({
                "Product": [cat.label(i) for i in recs],
                "Price": [som(cat.prices[i]) for i in recs],
                "Chance to accept": [f"{100 * chance[i]:.1f}%" for i in recs],
                "Business value": [som(int(round(value[i]))) for i in recs]}),
                hide_index=True, use_container_width=True)
            st.caption("**Business value** = chance to accept × price. Sorted automatically, best first. "
                       "Chance and value come from the cart only, so they do not change with shopper history.")
            if s == 0:
                st.caption("⚠ Price sensitivity is None: expensive products look as good as cheap ones, "
                           "so pricey items are favoured. Try Low or Medium.")
            st.caption(f"Expected extra revenue if all are shown: **{som(int(round(value[recs].sum())))}**.")