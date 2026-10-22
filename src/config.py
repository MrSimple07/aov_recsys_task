"""Central settings. Prices are converted to so'm once, at training time."""
from dataclasses import dataclass

# Raw dataset prices are in this currency (Online Retail II = British pounds).
SOURCE_CURRENCY = "GBP"
# Approximate rate. CHECK the Central Bank of Uzbekistan rate and edit, or pass --fx to
# train.py. This only relabels 2010-2011 UK prices; it is NOT an inflation adjustment.
FX_GBP_UZS = 16100.0
ROUND_TO = 100          # prices are rounded to the nearest 100 so'm (min 100) at training time
MAX_SLOTS = 10          # the UI / simulator never show more than 10 suggestions

# The app replays only SMALL orders, like an app basket (the raw data is wholesale: median 15
# products, mean 26). Orders with this many different products are used (cart = half of the order).
SIM_ORDER_ITEMS = (4, 8)
# Prices are rescaled when the app loads (no retraining) so that the average cart value, i.e. the
# AOV without suggestions, equals this. Change it to move all prices together; ratios,
# percentages and model rankings do not change. A 3-product cart then costs about 120 000 so'm.
TARGET_AOV = 120_000

# The Fixed list should be accepted about this often (real Bellissimo figure). The app
# tunes the shopper's "accept" chance automatically to land here.
TARGET_ACCEPT = 0.24
# Accept chance of an UNWANTED but popular product = MISS_RATIO x accept chance of a wanted one
# (obscure products are scaled down further, see Catalog.impulse). 0.10 gives: random products
# accepted by ~2% of shoppers, the Fixed list by 24%. This is an assumption and it matters: the
# higher it is, the more of the Fixed list's 24% is impulse and the less the smarter models gain.
MISS_RATIO = 0.0001
# Demo assumption: the monthly business value of the best model should land in this range (so'm).
MONTH_VALUE_RANGE = (1_000_000, 2_000_000)



def som(x: float) -> str:
    """Format a number as so'm with space-separated thousands: 1 234 500 so'm."""
    return f"{x:,.0f}".replace(",", " ") + " so'm"


@dataclass
class SimConfig:
    # What the business controls
    n_slots: int = 10                    # suggestions shown (1..10)
    price_limit: float = float("inf")    # so'm; pricier items are never suggested
    # Shopper assumptions (synthetic - there are no accept/reject logs in the data).
    # p_hit / p_miss are overwritten by the automatic calibration in the app.
    cart_frac: float = 0.5               # share of a real order already in the cart
    p_hit: float = 0.50                  # chance to accept an item the shopper wanted
    p_miss: float = 0.03                 # chance to accept an unrelated item
    price_sensitivity: float = 0.15      # accept chance *= exp(-s * price / median_price)
    position_bias: bool = True           # lower slots get less attention: 1/log2(rank+1)
    remove_hit: float = 0.05             # accepted wanted item is later removed
    remove_miss: float = 0.30            # accepted unrelated item is later removed
    # Orders that finish / stop (assumed - the data has no abandonment logs)
    finish_base: float = 0.80            # share of orders that finish if nothing is accepted
    stop_after_accept: float = 0.30      # avg share of shoppers who accept and then stop
    max_stop: float = 0.90