"""Bellissimo-style menu (Uzbek names, so'm prices). Replaces the Online Retail II products.

The raw data is a UK gift shop, so product names/prices are meaningless for a pizzeria.
`apply_menu(df)` keeps the real BASKET STRUCTURE (which products are bought together, when, by whom)
but renames every product to a menu item and sets its price from the menu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- pizzas: (name, price of 20 cm in so'm, popularity weight). Source: bellissimo.uz ("... dan") ----
PIZZAS = [
    ("Pepperoni", 37000, 5), ("Margarita", 34000, 3), ("Pishloqli joʻja", 33000, 3),
    ("Barbekyu tovuq", 33000, 3), ("Mr. Pep", 39000, 3), ("Kebab", 36000, 2.5),
    ("Alfredo", 37000, 2), ("Cheddar donar", 44000, 3), ("Donar pitsa", 44000, 2.5),
    ("Bellissimo", 44000, 2.5), ("Dabl pepperoni", 44000, 2), ("Kombo", 36000, 2),
    ("Texascha grill", 39000, 2), ("Ranch joʻjacha", 37000, 2), ("Halapeno", 36000, 1.5),
    ("Meksikancha", 41000, 1.5), ("Dabl chizburger", 39000, 1.5), ("Tovuqli donar", 39000, 2),
    ("Frimen", 39000, 1.5), ("Pishloqli pitsa", 29000, 1.5), ("Vegaboom", 36000, 1),
    ("Gavaycha", 33000, 1),
]
SIZES = {20: (1.00, 0.7), 25: (1.40, 1.0), 30: (2.30, 0.5)}      # cm: (price x, popularity x)

# ---- fixed-price items: (category, name, price so'm, weight). Prices marked ~ are ESTIMATES ----
OTHER = [
    ("Pitsa", "Goʻshtli miks (35 sm)", 109000, 1), ("Pitsa", "Toʻrt fasl (35 sm)", 109000, 1),
    ("Pitsa", "50 ga 50 (35 sm)", 109000, 1.5),
    ("Pitsa kruassan", "Kruassan Pepperoni (25 sm)", 89000, 1), ("Pitsa kruassan", "Kruassan Alfredo (25 sm)", 89000, 0.8),
    ("Pitsa kruassan", "Kruassan Pishloqli joʻja (25 sm)", 94000, 0.8), ("Pitsa kruassan", "Kruassan Margarita (25 sm)", 94000, 0.6),
    ("Pitsa kruassan", "Kruassan Barbekyu tovuq (25 sm)", 94000, 0.8), ("Pitsa kruassan", "Kruassan Cheddar donar (25 sm)", 99000, 0.8),
    # ~ snacks
    ("Gazaklar", "Pechdan chiqqan kartoshka fri", 19000, 6), ("Gazaklar", "Jaydari kartoshka", 19000, 3),
    ("Gazaklar", "Tovuq popperslari", 25000, 4), ("Gazaklar", "Tovuqli stripslar (3 ta)", 24000, 4),
    ("Gazaklar", "Tovuqli stripslar (5 ta)", 36000, 2), ("Gazaklar", "Tovuq qanotchalari (3 ta)", 24000, 2.5),
    ("Gazaklar", "Naggetslar (6 ta)", 23000, 1.5), ("Gazaklar", "Pishloqli tayoqchalar", 22000, 1.5),
    ("Gazaklar", "Chiabatta", 18000, 1), ("Gazaklar", "Bellisster", 20000, 1),
    # ~ drinks
    ("Ichimliklar", "Coca-Cola 0.5 L", 9000, 6), ("Ichimliklar", "Coca-Cola 1 L", 14000, 5),
    ("Ichimliklar", "Coca-Cola 1.5 L", 17000, 4), ("Ichimliklar", "Fanta 1 L", 14000, 2.5),
    ("Ichimliklar", "Sprite 1 L", 14000, 2.5), ("Ichimliklar", "Fanta 0.5 L", 9000, 1.5),
    ("Ichimliklar", "Sprite 0.5 L", 9000, 1.5), ("Ichimliklar", "Bliss Smile bolalar sharbati 220 ml", 8000, 1.5),
    ("Ichimliklar", "Red Bull 250 ml", 18000, 1.5), ("Ichimliklar", "Gazsiz suv 0.5 L", 5000, 2),
    ("Ichimliklar", "Qora choy", 6000, 1.5), ("Ichimliklar", "Koʻk choy", 6000, 1),
    ("Ichimliklar", "Amerikano", 15000, 1.5), ("Ichimliklar", "Kapuchino", 18000, 1.5),
    ("Ichimliklar", "Muzli choy 0.5 L", 12000, 1.5),
    # ~ salads
    ("Salatlar", "Sezar salati", 29000, 1.5), ("Salatlar", "Yunon salati", 26000, 1),
    ("Salatlar", "Tovuqli salat", 28000, 1), ("Salatlar", "Koleslo (karam salati)", 15000, 1),
    # ~ desserts
    ("Desertlar", "Sinnamonchalar (4 ta)", 22000, 3), ("Desertlar", "Sinnamonchalar (8 ta)", 39000, 1),
    ("Desertlar", "Chizkeyk", 24000, 1.5), ("Desertlar", "Shokoladli brauni", 18000, 1.5),
    ("Desertlar", "Tiramisu", 26000, 1), ("Desertlar", "Shokoladli fondan", 22000, 1),
    ("Desertlar", "Muzqaymoq", 12000, 1),
    # ~ sauces
    ("Souslar", "Pishloqli sous", 4000, 3), ("Souslar", "Barbekyu sousi", 3000, 3),
    ("Souslar", "Maxsus sous", 3000, 3), ("Souslar", "Ranch sousi", 3000, 2.5),
    ("Souslar", "Sarimsoqli sous", 3000, 2.5), ("Souslar", "Burger sousi", 3000, 1.5),
    ("Souslar", "Chili sous", 3000, 1.5), ("Souslar", "Ketchup", 2000, 2),
]


def build_menu() -> pd.DataFrame:
    """code, category, name, price (so'm), weight - sorted from most to least popular."""
    rows = []
    for name, p20, w in PIZZAS:
        for cm, (pm, wm) in SIZES.items():
            rows.append((f"pizza-{len(rows)}", "Pitsa", f"{name} ({cm} sm)",
                         int(round(p20 * pm / 1000) * 1000), w * wm))
    rows += [(f"item-{i}", c, n, p, w) for i, (c, n, p, w) in enumerate(OTHER)]
    m = pd.DataFrame(rows, columns=["code", "category", "name", "price", "weight"])
    return m.sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)


def apply_menu(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Map every raw product to a menu item (columns item / desc / price are replaced).

    The most popular raw products map 1:1 to menu items in popularity order. The long tail
    (thousands of rare gift-shop products) is folded into menu items at random, weighted by
    the item's popularity weight, so baskets keep their shape but never show a gift-shop product.
    """
    menu = build_menu()
    rank = df.drop_duplicates(["invoice", "item"])["item"].value_counts().index
    n = len(menu)
    mapping = {it: i for i, it in enumerate(rank[:n])}
    tail = rank[n:]
    w = menu["weight"].to_numpy(float)
    pick = np.random.default_rng(seed).choice(n, size=len(tail), p=w / w.sum())
    mapping.update(dict(zip(tail, pick.tolist())))
    idx = df["item"].map(mapping).to_numpy(int)
    out = df.copy()
    out["item"] = menu["code"].to_numpy()[idx]
    out["desc"] = menu["name"].to_numpy()[idx]
    out["price"] = menu["price"].to_numpy(float)[idx]
    return out
