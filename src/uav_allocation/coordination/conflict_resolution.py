"""Deterministic total order over competing claims on one task.

Rank = (phase, bid, -uav_id) compared lexicographically:
  1. phase   : EXECUTING > CONFIRMED > PROVISIONAL  (work already under way is never pre-empted)
  2. bid     : higher local utility wins (rounded to 6 d.p. so every node compares identical numbers)
  3. uav id  : lower id wins ties -> the order is total, so all nodes that see the same two claims agree.
"""
from __future__ import annotations

from typing import Tuple

from .reservations import Reservation


def priority_key(r: Reservation) -> Tuple[int, float, int]:
    return (int(r.status), round(float(r.bid), 6), -int(r.uav_id))


def winner(a: Reservation, b: Reservation) -> Reservation:
    return a if priority_key(a) >= priority_key(b) else b
