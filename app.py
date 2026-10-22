"""Recommender lab - which model earns more?   Run:  streamlit run app.py"""
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import inspect

import altair as alt
import pandas as pd
import streamlit as st

from src.artifacts import load_artifacts
from src.config import TARGET_ACCEPT, TARGET_AOV, SimConfig, som
from src.explain import why_shown
from src.experiment import calibrate_accept, calibrate_relevance, prepare_bundle, estimate_stop_slope, make_policies, run_simulation
from src.models import Ctx, accept_prob, relevance
from src.simulator import _subsample, build_sessions, run_policy, simulate_cart
from src.config import MONTH_VALUE_RANGE, TARGET_ACCEPT, TARGET_AOV, SimConfig, som
from src.config import AVG_ORDER_SOM, ORDERS_PER_MONTH, TARGET_ACCEPT, TARGET_AOV, SimConfig, som

import pandas as pd

st.set_page_config(page_title="Which recommender earns more?", page_icon="🛒", layout="wide")
ART = os.environ.get("AOV_ARTIFACTS", "artifacts")
ORDERS, REPEATS = 1500, 3        # fixed: steady enough, fast enough

if not (Path(ART) / "meta.json").exists():
    st.error("No trained models found. Train once, then reload this page:")
    st.code("python train.py --data data/raw", language="bash")
    st.stop()

BLURB = {
    "Random": "Random products. The floor every model must beat.",
    "Fixed list": "The same best sellers for everyone.",
    "Association rules": "Products often bought together with the cart.",
    "Sequential": "Next product from the shopper's past orders and cart.",
    "Contextual bandit (cold start)": "Learns from accepted suggestions while running. Starts knowing nothing.",
    "Contextual bandit (warm start)": "Same learner, but pre-trained on past orders first.",
}

ACCENT = "#1b7f3b"
# old Streamlit: use_container_width=True; new Streamlit: width="stretch"
STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters
           else {"use_container_width": True})
STRETCH_CHART = ({"width": "stretch"} if "width" in inspect.signature(st.altair_chart).parameters
                 else {"use_container_width": True})
st.markdown(f"""<style>
div[data-testid="stTabs"] [role="tablist"]{{gap:12px;border-bottom:3px solid {ACCENT};margin-bottom:14px}}
div[data-testid="stTabs"] [role="tab"]{{height:auto;padding:14px 34px;border:2px solid {ACCENT};cursor:pointer;
 border-bottom:none;border-radius:14px 14px 0 0;background:rgba(128,128,128,.12)}}
div[data-testid="stTabs"] [role="tab"] p{{font-size:1.3rem;font-weight:700;margin:0}}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{background:{ACCENT}}}
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p{{color:#fff}}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],div[data-testid="stTabs"] [data-baseweb="tab-border"],
div[data-testid="stTabs"] .react-aria-SelectionIndicator{{display:none}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_bundle(path):
    b = load_artifacts(path)
    b.prep = prepare_bundle(b, rescale=not b.meta.get("menu"))           # small orders only + prices scaled to the target AOV, once
    return b


@st.cache_data(show_spinner=False)
def simulate(path, cfg_items, fixed_items):
    return run_simulation(get_bundle(path), SimConfig(**dict(cfg_items)), REPEATS, ORDERS,
                          list(fixed_items))


@st.cache_data(show_spinner=False)
def calibration(path, fixed_items):
    return calibrate_accept(get_bundle(path), list(fixed_items))


@st.cache_resource(show_spinner=False)
def live_models(path, cfg_items, fixed_items):
    """Models for the cart tab; the warm-start bandit is pre-trained on past orders."""
    b = get_bundle(path)
    cfg = SimConfig(**dict(cfg_items))
    pols = make_policies(b, cfg, list(fixed_items))
    slope = estimate_stop_slope(b, cfg)
    rel_scale = calibrate_relevance(b, cfg)
    sess = build_sessions(_subsample(list(b.warm), 1500), b.cat.prices, cfg, 7, b.ctxb)
    for p in pols:
        p.reset(0)
        if p.warm_start and sess:
            run_policy(p, sess, b.cat, cfg, slope)
    return {p.name: p for p in pols}, slope, rel_scale


b = get_bundle(ART)
cat, rules, meta = b.cat, b.rules, b.meta


def nm(i) -> str:
    return " ".join(cat.label(int(i)).split()).replace(" ,", ",")




def fmt(x, sign=False) -> str:
    t = f"{x:+,.0f}" if sign else f"{x:,.0f}"
    return t.replace(",", " ")


# ------------------------------------------------------------------------------ header
st.title("🛒 Which recommender raises the order value (AOV) the most?")
st.caption("AOV = average value of a finished order. We replay real orders, let each model suggest products "
           "at checkout and simulate whether shoppers accept. Money is in so'm. Shopper behaviour is assumed, not measured.")

# ------------------------------------------------------------------------------ settings
pmin, pmax = int(cat.prices.min()), int(cat.prices.max())
step = 1000 if pmax > 100_000 else 100
default_limit = int(min(max(round(np.quantile(cat.prices, 0.9), -3), pmin), pmax))
labels = {f"{nm(i)} · {som(cat.prices[i])}": int(i) for i in cat.pop_order[:300]}

with st.form("settings"):
    c1, c2 = st.columns(2)
    n_slots = c1.slider("Suggestions shown", 1, 10, 10)
    c1.caption("Products shown at checkout (up to 10).")
    limit = c2.slider("Highest price allowed (so'm)", pmin, pmax, default_limit, step)
    no_limit = c2.checkbox("No price limit", value=False)
    c2.caption("Pricier products are never suggested.")
    with st.expander("Fixed list products (optional)"):
        fixed_sel = st.multiselect("Shown to everyone in the 'Fixed list' model", list(labels),
                                   default=list(labels)[:10], max_selections=20)
        st.caption("Default: the 10 best sellers.")
    st.form_submit_button("Update results", type="primary")

s = 0.15        # price sensitivity: Low (fixed, not in the UI)
leave = 0.25    # leave risk after accepting: 25% (fixed, not in the UI)
fixed_items = tuple(labels[x] for x in fixed_sel) or tuple(int(i) for i in cat.pop_order[:10])
with st.spinner("Calibrating shopper behaviour..."):
    p_hit, p_miss, reached = calibration(ART, fixed_items)
cfg = SimConfig(n_slots=n_slots, price_limit=float("inf") if no_limit else float(limit),
                price_sensitivity=s, p_hit=p_hit, p_miss=p_miss, stop_after_accept=leave)
cfg_items = tuple(sorted(asdict(cfg).items()))

note = f"Shoppers are tuned so {100 * TARGET_ACCEPT:.0f}% of them accept at least one Fixed list suggestion (default settings)"
note += f" (reached {100 * reached:.0f}%)." if reached < TARGET_ACCEPT - 0.01 else "."
st.caption(note + f" 80% of orders finish when nothing is accepted; after accepting, about {100 * leave:.0f}% stop, "
           "more often when the add-on is pricey for the cart (assumptions).")


tab_cmp, tab_cart = st.tabs(["📊  Compare models", "🧺  Try a cart"])

# ---------------------------------------------------------------------------- compare tab
with tab_cmp:
    with st.spinner("Simulating all models (a few seconds)..."):
        summary, bands = simulate(ART, cfg_items, fixed_items)
    months = max((pd.Timestamp(meta["date_max"]) - pd.Timestamp(meta["date_min"])).days / 30.44, 1.0)
    orders_est = meta["orders"] * b.prep["share"] / months       # orders of the simulated size
    orders_pm = ORDERS_PER_MONTH                                  # 150 000 orders / month
    month_rev0 = orders_pm * AVG_ORDER_SOM                         # 18 000 000 000 so'm / month
    summary["month_extra"] = month_rev0 * summary["RPS_uplift_pct"] / 100   # extra revenue = revenue x uplift %
    aov0 = float(summary["AOV_base"].iloc[0])
    best = summary.loc[summary["AOV_uplift_pct"].idxmax()]
    m1, m2, m4, m3 = st.columns(4)
    m1.metric("AOV without suggestions", som(aov0))
    m1.caption("Average value of a finished order (cart counts 1 unit per product).")
    m2.metric(f"Best model: {best['policy']}", som(best["AOV_policy"]),
              f"{fmt(best['extra_per_order'], True)} so'm per order ({best['AOV_uplift_pct']:+.1f}%)",
              delta_color="off")
    m2.caption("AOV with suggestions.")
    m4.metric("Business value, so'm / month", fmt(best["month_extra"], True),
              f"{best['RPS_uplift_pct']:+.1f}% monthly revenue", delta_color="off")
    m4.caption(f"Extra revenue per month with {best['policy']}, after shoppers who leave.")
    fl = summary[summary["policy"] == "Fixed list"].iloc[0]
    m3.metric("Fixed list: shoppers who accept", f"{100 * fl['orders_with_accept']:.1f}%")
    m3.caption(f"Share of shoppers accepting at least one suggestion. Tuned to "
               f"{100 * TARGET_ACCEPT:.0f}% at default settings; moves with your settings.")

    if best["AOV_uplift_pct"] > best["AOV_uplift_pct_ci95"] and best["AOV_uplift_pct"] > 0.5:
        st.success(f"**{best['policy']}** raises AOV the most: {fmt(best['extra_per_order'], True)} so'm per "
                   f"order ({best['AOV_uplift_pct']:+.1f}%).")
    else:
        st.warning("No model clearly raises AOV here. Try a lower price limit or lower price sensitivity.")
    if best["RPS_uplift_pct"] < 0:
        st.info(f"Note: with {best['policy']} AOV rises but monthly revenue changes by "
                f"{best['RPS_uplift_pct']:+.1f}% because shoppers leave.")

    tbl = pd.DataFrame({
        "Model": summary["policy"],
        "What it does": summary["policy"].map(BLURB),
        "AOV (so'm)": summary["AOV_policy"],
        "Change (so'm)": summary["extra_per_order"],
        "Change (%)": summary["AOV_uplift_pct"],
        "Accept (%)": 100 * summary["orders_with_accept"],
        "Completed (%)": 100 * summary["finish_rate"],
        "Extra revenue / month (so'm)": summary["month_extra"],
        "Monthly revenue (%)": summary["RPS_uplift_pct"]})
    sty = (tbl.style.format({"AOV (so'm)": lambda v: fmt(v), "Change (so'm)": lambda v: fmt(v, True),
                             "Change (%)": "{:+.1f}%", "Accept (%)": "{:.1f}%",
                              "Completed (%)": "{:.1f}%",
                             "Extra revenue / month (so'm)": lambda v: fmt(v, True),
                             "Monthly revenue (%)": "{:+.1f}%"})
           .highlight_max(subset=["AOV (so'm)", "Change (so'm)", "Change (%)",
                                  "Extra revenue / month (so'm)", "Monthly revenue (%)"],
                          props=f"background-color:{ACCENT};color:#ffffff;font-weight:600"))
    st.dataframe(sty, hide_index=True, **STRETCH,
                 column_config={"Model": st.column_config.TextColumn(width="medium"),
                                "What it does": st.column_config.TextColumn(width="medium")})
    st.caption("**AOV** = average value of finished orders, with suggestions; **Change** = vs no suggestions. "
               "**Accept** = share of shoppers who accept at least one suggestion. "
               "**Completed** = share of started orders that finish; it falls "
               "when accepted add-ons scare shoppers away. **Business value** = extra revenue per month and its share "
               f"of monthly revenue (around {fmt(month_rev0)} so'm without suggestions at {orders_pm:,} orders). Revenue "
               "counts every started order, so it can grow less than AOV, or fall. Green = best. Random is the noise floor.")
    st.download_button("Download table (CSV)", summary.to_csv(index=False), "model_comparison.csv")

    st.markdown("**AOV change vs no suggestions (%)**")
    st.altair_chart(alt.Chart(tbl[["Model", "Change (%)"]]).mark_bar(color=ACCENT).encode(
        x=alt.X("Model:N", sort=None, axis=alt.Axis(labelAngle=0, labelLimit=150, title=None)),
        y=alt.Y("Change (%):Q", title="% change in AOV"), tooltip=["Model", "Change (%)"]),
        **STRETCH_CHART)

    st.subheader("Price bands: where the AOV comes from")
    st.caption("Cheap products are accepted more often, but value = chance to accept × price. A product "
               "accepted 70% of the time at 5 adds 3.5; one accepted 30% at 30 adds 9.")
    names = list(summary["policy"])
    pick = st.selectbox("Model", names, index=names.index(best["policy"]), key="band_model")
    bb = bands[bands["policy"] == pick].set_index("band")
    cc = st.columns(2)
    cc[0].markdown("**Chance a shown suggestion is accepted (%)**")
    cc[0].bar_chart(100 * bb["conversion"], color=ACCENT)
    cc[1].markdown("**Order value added per suggestion shown (so'm)**")
    cc[1].bar_chart(bb["value_per_suggestion"], color=ACCENT)

# -------------------------------------------------------------------------------- cart tab
with tab_cart:
    st.caption("Build a cart and see what each model would show. Products are listed exactly as the shopper "
               "sees them (the model's best pick first). Click a column header to sort.")
    model = st.selectbox("Model", list(BLURB), index=2, key="cart_model")
    st.caption(BLURB[model])
    opts = {f"{nm(i)} · {som(cat.prices[i])}": int(i) for i in cat.pop_order[:300]}
    picked = st.multiselect("Products in the cart", list(opts), max_selections=15, default=list(opts)[20:23])
    cart = np.array([opts[x] for x in picked], dtype=int)

    hist = b.seq.history
    who = st.radio("Shopper", ["New shopper (no history)", "Returning shopper"], horizontal=True)
    customer = -1
    if who.startswith("Returning"):
        if not hist:
            st.info("No customer histories were saved with the models.")
        else:
            ids = sorted(hist, key=lambda c: -len(hist[c]))[:200]
            customer = st.selectbox("Customer", ids,
                                    format_func=lambda c: f"Customer {c} · {len(hist[c])} past items")
            st.caption("Recently bought: " + ", ".join(nm(i) for i in hist[customer][-6:]))
            if model != "Sequential":
                st.caption("ℹ Only the **Sequential** model uses purchase history.")

    if len(cart) == 0:
        st.info("Add at least one product to the cart.")
    else:
        with st.spinner("Preparing models..."):
            live, slope, rel_scale = live_models(ART, cfg_items, fixed_items)
        B = float(cat.prices[cart].sum())
        ts = time.time_ns()
        rel = np.clip(rel_scale * relevance(rules, cat, cart), 0.0, 1.0)   # same scale as the simulation
        shown = {}
        for name, p in live.items():                      # every model, once, so tables agree
            vec = p.ctxb.build(cart, ts) if hasattr(p, "ctxb") else None
            ctx = Ctx(customer, ts, vec)
            shown[name] = ([int(i) for i in p.recommend(cart, n_slots, cfg.price_limit, ctx)], ctx)
        recs, ctx = shown[model]
        pol = live[model]

        with st.expander(f"Cart: {len(cart)} products · {som(B)}"):
            st.dataframe(pd.DataFrame({"Product": [nm(i) for i in cart],
                                       "Price (so'm)": [cat.prices[i] for i in cart]})
                         .style.format({"Price (so'm)": lambda v: fmt(v)}), hide_index=True, **STRETCH)
        if model == "Contextual bandit (cold start)":
            st.caption("The cold-start bandit has learned nothing here (it only learns during the simulation), "
                       "so its picks are close to random.")

        if not recs:
            st.write("Nothing to suggest under this price limit.")
        else:
            m = len(recs)
            price = cat.prices[recs]
            r = rel[recs]
            pos = 1.0 / np.log2(np.arange(m) + 2) if cfg.position_bias else np.ones(m)
            pen = np.exp(-cfg.price_sensitivity * price / cat.ref)
            chance = (r * cfg.p_hit + (1 - r) * cfg.p_miss) * pen * pos
            keep = 1.0 - (r * cfg.remove_hit + (1 - r) * cfg.remove_miss)
            add = chance * keep * price
            leave_if = 100 * np.minimum(slope * price / B, cfg.max_stop - (1 - cfg.finish_base))
            why = why_shown(pol, cat, cart, recs, ctx, customer)

            sim = simulate_cart(cat, rel, cart, recs, cfg, slope)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Cart value now", som(B))
            k2.metric("AOV with this list", som(sim["aov_after"]))
            k3.metric("AOV change", f"{sim['aov_change_pct']:+.1f}%", f"{fmt(sim['aov_change'], True)} so'm",
                      delta_color="off")
            k4.metric("Shoppers who accept", f"{100 * sim['accept_share']:.0f}%")
            k5.metric("Orders completed", f"{100 * sim['finish_rate']:.1f}%")

            df = pd.DataFrame({
                "Slot": np.arange(1, m + 1),
                "Product": [nm(i) for i in recs],
                "Price (so'm)": price,
                "Why shown": why,
                "Chance (%)": 100 * chance,
                "Price penalty (%)": 100 * (1 - pen),

                "Order grows (%)": 100 * add / B,
                "Leave risk (%)": leave_if,
                "Exp. add (score)": add,
                
                })
            st.dataframe(df.style.format({
                "Price (so'm)": lambda v: fmt(v), "Chance (%)": "{:.1f}%",
                "Price penalty (%)": "{:.0f}%", "Exp. add (so'm)": lambda v: fmt(v),
                "Order grows (%)": "+{:.1f}%", "Leave risk (%)": "+{:.1f}%"}),
                hide_index=True, **STRETCH,
                         column_config={"Slot": st.column_config.NumberColumn(width="small"),
                                        "Why shown": st.column_config.TextColumn(width="large")})
            st.caption("**Chance** (to accept) includes the price penalty and the slot (lower slots get less attention), "
                       "so the same product gets a different chance under different models. **Exp. add** = "
                       "chance × price × (1 − removal risk). **Order grows** = expected add ÷ cart value. **Leave "
                       "risk** = extra shoppers who stop if they accept it. **AOV with this list** replays this "
                       "cart thousands of times with the same shopper rules as the comparison tab.")
            if cfg.price_sensitivity == 0:
                st.caption("⚠ Price sensitivity is None: expensive products look as good as cheap ones.")
            if sim["revenue_change_pct"] < 0 <= sim["aov_change_pct"]:
                st.warning(f"AOV rises, but revenue per started order changes by {sim['revenue_change_pct']:+.1f}% "
                           "because shoppers leave. Try a lower price limit.")

        st.subheader("Same cart, every model")
        rows = []
        for name, (rc, _) in shown.items():
            sm = simulate_cart(cat, rel, cart, rc, cfg, slope)
            rows.append({"Model": name, "Products shown": len(rc),
                         "AOV with suggestions (so'm)": sm["aov_after"],
                         "AOV change (%)": sm["aov_change_pct"],
                         "Shoppers who accept (%)": 100 * sm["accept_share"],
                         "Orders completed (%)": 100 * sm["finish_rate"]})
        allm = pd.DataFrame(rows)
        st.dataframe(allm.style.format({"AOV with suggestions (so'm)": lambda v: fmt(v),
                                        "AOV change (%)": "{:+.1f}%", "Shoppers who accept (%)": "{:.0f}%",
                                        "Orders completed (%)": "{:.1f}%"})
                     .highlight_max(subset=["AOV change (%)"],
                                    props=f"background-color:{ACCENT};color:#ffffff;font-weight:600"),
                     hide_index=True, **STRETCH)
        st.caption("One cart gives noisy results; the Compare tab averages over many real orders.")
