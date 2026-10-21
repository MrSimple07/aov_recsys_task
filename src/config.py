"""Central settings. Prices are converted to so'm once, at training time."""
from dataclasses import dataclass

# Raw dataset prices are in this currency (Online Retail II = British pounds).
SOURCE_CURRENCY = "GBP"
# Approximate rate (Aug-Oct 2026 sources: ~16,000-16,600). CHECK the Central Bank
# of Uzbekistan rate and edit, or pass --fx to train.py. Note: this only relabels
# 2010-2011 UK prices; it is NOT an inflation / purchasing-power adjustment.
FX_GBP_UZS = 16100.0
ROUND_TO = 100          # prices are rounded to the nearest 100 so'm (min 100)
MAX_SLOTS = 10          # the UI / simulator never show more than 10 suggestions


def som(x: float) -> str:
    """Format a number as so'm with space-separated thousands: 1 234 500 so'm."""
    return f"{x:,.0f}".replace(",", " ") + " so'm"


@dataclass
class SimConfig:
    # What the business controls
    n_slots: int = 10                    # suggestions shown (1..10)
    price_limit: float = float("inf")    # so'm; pricier items are never suggested
    # Shopper assumptions (synthetic - there are no accept/reject logs in the data)
    cart_frac: float = 0.5               # share of a real order already in the cart
    p_hit: float = 0.50                  # chance to accept an item the shopper wanted
    p_miss: float = 0.03                 # chance to accept an unrelated item
    price_sensitivity: float = 0.15      # accept chance *= exp(-s * price / median_price)
    position_bias: bool = True           # lower slots get less attention: 1/log2(rank+1)
    remove_hit: float = 0.05             # accepted wanted item is later removed
    remove_miss: float = 0.30            # accepted unrelated item is later removed
    # Orders that finish / stop (synthetic)
    finish_base: float = 0.80            # share of orders that finish if nothing is accepted
    stop_after_accept: float = 0.30      # avg share of shoppers who accept and then stop
    max_stop: float = 0.90
