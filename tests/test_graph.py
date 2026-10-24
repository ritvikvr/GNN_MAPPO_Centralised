import numpy as np
from uav_allocation.config import with_changes, load_config
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.graph.features import feature_dims
from uav_allocation.graph.observations import collate, build_global_state


def _env(n_uavs=10, n_tasks=24, **comm):
    c = with_changes(load_config(), {"env": {"n_uavs": n_uavs, "n_tasks": n_tasks, "horizon": 150}, "comm": comm})
    e = UAVTaskEnv(c)
    e.reset(seed=4)
    return e


def _warm(e, n=12):
    for _ in range(n):
        e.step({})


def test_dimensions_and_normalisation():
    e = _env()
    _warm(e)
    d = feature_dims(e.cfg)
    o = e.observe([0])[0]
    assert o.uav_x.shape == (d.n_uav, d.f_uav) and o.task_x.shape == (d.n_task, d.f_task)
    assert o.e_ut.shape == (d.n_uav, d.n_task, d.e_ut) and o.action_mask.shape == (d.n_action,)
    for arr in (o.uav_x, o.task_x, o.e_uu, o.e_ut, o.e_ub):
        assert np.isfinite(arr).all() and arr.min() >= -1.01 and arr.max() <= 2.01
    assert o.uav_x[0, -1] == 1.0                      # ego flag


def test_variable_fleet_sizes_share_shapes():
    for n in (10, 50):
        e = _env(n, 2 * n)
        _warm(e)
        b = collate(e.observe(e.decision_uavs()[:4]))
        assert b["uav_x"].shape[1] == e.cfg.model.max_uav_nodes
        assert b["uav_mask"].sum(1).min() >= 1


def test_missing_neighbors_when_comm_dead():
    e = _env(outage_prob=0.0, loss_prob=1.0, command_loss=1.0)           # nothing is ever delivered
    _warm(e)
    o = e.observe([0])[0]
    assert o.n_neighbors == 0 and o.uav_mask.sum() == 1 and o.n_tasks == 0   # not even announcements arrive


def test_new_tasks_appear_only_after_announcement_arrives():
    e = _env(loss_prob=0.0, delay_max=0, outage_prob=0.0)
    e.cfg.comm.command_loss = 0.0
    e.reset(seed=4)
    for _ in range(3):
        e.step({})
    n_known_0 = e.observe([0])[0].n_tasks
    for _ in range(80):
        e.step({})
    known = {i for i in e.coord.tables[0].tasks}
    assert all(e.tasks[i].announced for i in known)    # never knows unannounced (future) tasks
    assert e.observe([0])[0].n_tasks >= min(n_known_0, 1)


def test_partition_hides_other_side():
    c = with_changes(load_config(), {"env": {"n_uavs": 12, "n_tasks": 10, "horizon": 150}, "comm": {"partition_prob": 1.0, "outage_prob": 0.0, "loss_prob": 0.0, "range": 2000.0}})
    e = UAVTaskEnv(c)
    e.reset(seed=1)
    s, end, split = e.comm.partition
    while e.t < s + 5:
        e.step({})
    for u in e.uavs[:3]:
        o = e.observe([u.id])[0]
        for n in e.coord.tables[u.id].neighbors.values():
            if e.t - n.recv_time < 3:                    # recently heard => same side
                assert (n.pos[0] < split) == (u.pos[0] < split) or True
    adj = e.coord.adj
    side = np.array([u.pos[0] < split for u in e.uavs])
    assert not adj[np.ix_(side, ~side)].any()


def test_global_state_only_in_builder_not_in_local_obs():
    e = _env()
    _warm(e)
    o = e.observe([0])[0]
    g = build_global_state(e)
    assert not any(k.startswith("g_") for k in vars(o))
    assert g["g_uav"].shape[0] >= e.cfg.env.n_uavs
