"""Local bid (utility) used to rank competing claims. Pure function of locally known quantities."""
from __future__ import annotations


def compute_bid(priority: float, eta: float, service: float, slack: float, e_travel: float, capacity: float) -> float:
    """Value per second of effort, boosted by deadline urgency and discounted by energy cost.

    priority : mission value (1..5)       eta     : nominal travel time (s)
    service  : service duration (s)       slack   : deadline - (now + eta + service) (s)
    e_travel : travel energy              capacity: battery capacity (same units)
    """
    urgency = 1.0 + 1.0 / (1.0 + max(slack, 0.0) / 60.0)
    energy_pen = 1.0 + 0.5 * (e_travel / max(capacity, 1e-9))
    return float(priority * urgency / ((eta + 0.5 * service + 5.0) * energy_pen))
