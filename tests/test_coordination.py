import itertools

import numpy as np

from uav_allocation.coordination.conflict_resolution import winner
from uav_allocation.coordination.reservations import LocalReservationTable, Outcome, ResStatus, Reservation
from uav_allocation.environment.task import TaskInfo

IDS = itertools.count(1)
INFO = TaskInfo(0, 1, 1, 0, 3, 0, 1000, 10, 0)


def tab(i):
    t = LocalReservationTable(i, lease_duration=12, confirm_window=3, renew_period=4)
    t.learn_task(INFO, 0)
    return t


def test_total_order_and_tiebreak():
    a = Reservation(0, 1, 2.0, 0, ResStatus.PROVISIONAL, 0, 12, 1)
    b = Reservation(0, 2, 2.0, 0, ResStatus.PROVISIONAL, 0, 12, 2)
    assert winner(a, b).uav_id == 1 and winner(b, a).uav_id == 1       # equal bids -> lower id, symmetric
    c = Reservation(0, 3, 0.1, 0, ResStatus.CONFIRMED, 0, 12, 3)
    assert winner(a, c).uav_id == 3                                    # committed work is never pre-empted by a better bid


def test_simultaneous_requests_trigger_conflict_resolution_and_converge():
    A, B = tab(1), tab(2)
    ra = A.request(0, 5.0, 10, next(IDS))
    rb = B.request(0, 3.0, 10, next(IDS))
    assert A.receive_claim(rb, 11) == Outcome.KEPT                    # A outranks B
    assert B.receive_claim(ra, 11) == Outcome.LOST_OWN                # B must yield
    assert B.own is None and A.own is not None
    assert B.claims[0].uav_id == 1 and A.claims[0].uav_id == 1        # both views agree


def test_not_exclusive_until_confirmed_then_confirms_after_window():
    A = tab(1)
    A.request(0, 1.0, 10, next(IDS))
    assert A.own.status == ResStatus.PROVISIONAL
    recs, ev = A.tick(11, 0, IDS)
    assert not ev
    recs, ev = A.tick(13, 0, IDS)
    assert ("confirmed", 0) in ev and A.own.status == ResStatus.CONFIRMED
    A2 = tab(1)
    A2.request(0, 1.0, 10, next(IDS))
    _, ev = A2.tick(13, link_ok_since=12, msg_ids=IDS)               # radio was down during the window
    assert ("confirmed", 0) not in ev


def test_stale_messages_rejected_and_release_bumps_version():
    A, B = tab(1), tab(2)
    r = A.request(0, 1.0, 0, next(IDS))
    assert B.receive_claim(r, 1) == Outcome.ACCEPTED
    rel = A.release_own(5, False, next(IDS))
    assert B.receive_claim(rel, 6) == Outcome.RELEASED and B.epoch(0) == 1
    assert B.receive_claim(r, 7) in (Outcome.STALE, Outcome.DUPLICATE)   # delayed old claim must not resurrect the task
    old = Reservation(0, 3, 9.0, 0, ResStatus.CONFIRMED, 2, 14, next(IDS))
    assert B.receive_claim(old, 8) == Outcome.STALE
    expired = Reservation(0, 3, 9.0, 1, ResStatus.CONFIRMED, 2, 4, next(IDS))
    assert B.receive_claim(expired, 20) == Outcome.STALE


def test_expired_lease_allows_safe_reassignment_and_old_owner_yields_on_return():
    A, B = tab(1), tab(2)
    r = A.request(0, 5.0, 0, next(IDS))
    A.tick(3, 0, IDS)
    B.receive_claim(A.own, 4)
    assert B.claims[0].uav_id == 1
    _, ev = B.tick(30, 0, IDS)                                         # A went silent (failed / partitioned)
    assert ("lease_expired", 0) in ev and 0 not in B.claims
    rb = B.request(0, 2.0, 31, next(IDS))                              # task reassigned to B
    B.tick(34, 0, IDS)
    assert B.own.status == ResStatus.CONFIRMED
    # A reappears (still alive, still believes it owns the task) and renews its lease: contention is settled by the
    # deterministic order, so both nodes converge on the same winner (A: equal phase, higher bid).
    recs, _ = A.tick(40, 0, IDS)
    assert recs and recs[0].status == ResStatus.CONFIRMED
    assert B.receive_claim(recs[0], 41) == Outcome.LOST_OWN and B.own is None
    assert A.receive_claim(B.claims.get(0) or recs[0], 41) in (Outcome.DUPLICATE, Outcome.KEPT)
    assert A.own is not None


def test_partition_heal_gossip_resolves_double_confirmation():
    A, B, C = tab(1), tab(2), tab(3)
    A.request(0, 5.0, 0, next(IDS)); A.tick(3, 0, IDS)
    B.request(0, 4.0, 0, next(IDS)); B.tick(3, 0, IDS)                 # both confirmed during partition
    assert A.own.status == B.own.status == ResStatus.CONFIRMED
    # heal: C relays what it hears; losers learn from gossip too
    C.receive_claim(A.own, 10); C.receive_claim(B.own, 10)
    assert C.claims[0].uav_id == 1
    assert B.receive_claim(C.claims[0], 11) == Outcome.LOST_OWN
    assert A.receive_claim(B.own if B.own else C.claims[0], 11) in (Outcome.DUPLICATE, Outcome.KEPT)
    assert A.own is not None and B.own is None


def test_network_protocol_end_to_end_conflict():
    from uav_allocation.config import load_config, with_changes
    from uav_allocation.environment.communication import CommNetwork
    from uav_allocation.coordination.protocol import CoordinationLayer
    cfg = with_changes(load_config(), {"comm": {"loss_prob": 0.0, "delay_min": 0, "delay_max": 0, "outage_prob": 0.0}})
    comm = CommNetwork(cfg.comm, 3, np.random.default_rng(0))
    co = CoordinationLayer(3, cfg.coordination, cfg.comm, comm)
    pos = np.array([[0, 0], [10, 0], [20, 0]], float)
    alive = np.ones(3, bool)
    co.begin_step(0, pos, alive)
    co.request(0, INFO, 5.0, 0)
    co.request(1, INFO, 3.0, 0)
    events = []
    for t in range(1, 8):
        co.begin_step(t, pos, alive)
        events += co.deliver(t, alive)
        events += co.tick(t, comm.outage_until, alive)
    assert (1, "yield", 0) in events and (0, "yield", 0) not in events
    assert co.tables[0].own is not None and co.tables[1].own is None
    assert co.metrics["conflicts"] >= 1
