from dataclasses import dataclass


@dataclass
class SimConfig:
    """Assumptions of the user-response simulator (see README)."""
    cart_frac: float = 0.5          # share of the final basket already in the cart when we recommend
    p_hit: float = 0.5              # accept prob. if the item is one the user latently wanted
    p_miss: float = 0.03            # accept prob. for an unrelated item (impulse buy)
    price_sensitivity: float = 0.15  # acceptance decays as exp(-s * price / median_price)
    slots: int = 1                  # number of recommendation slots shown
