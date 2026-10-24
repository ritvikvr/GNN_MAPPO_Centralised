import numpy as np

from uav_allocation.config import load_config, with_changes
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.environment.uav import Action, ActionKind, UAVStatus
from uav_allocation.graph.features import feature_dims
from uav_allocation.graph.observations import collate
from uav_allocation.models.graph_actor import GraphActor
from uav_allocation.safety import feasibility as F
from uav_allocation.safety.feasibility import TaskArrays
from uav_allocation.environment.task import TaskInfo


def _env(**env):
    c = with_changes(load_config(), {"env": {"n_uavs": 6, "n_tasks": 12, "horizon": 150, "initial_task_frac": 1.0, **env}})
    e = UAVTaskEnv(c)
    e.reset(seed=3)
    for _ in range(8):
        e.step({})
    return c, e


def _info(**k):
    base = dict(id=0, x=500, y=500, required_capability=0, priority=3, arrival_time=0, deadline=1e6, service_duration=10, payload=1)
    base.update(k)
    return TaskInfo(**base)


def test_each_reason_is_detected():
    c, e = _env()
    u = e.uavs[0]
    u.capabilities = np.array([True, False, False])
    ck = e.checker
    arr = TaskArrays.from_infos([_info(required_capability=1), _info(payload=99), _info(x=5000), _info(deadline=1.0), _info(service_duration=5000)])
    f = ck.check(u, arr, e.t)
    assert f.reasons[0] & F.CAPABILITY and f.reasons[1] & F.PAYLOAD and f.reasons[2] & F.GEOFENCE
    assert f.reasons[3] & F.DEADLINE and f.reasons[4] & F.ENERGY and not f.mask.any()


def test_no_fly_zone_blocks_path():
    c, e = _env(no_fly_zones=[[500.0, 500.0, 100.0]])
    u = e.uavs[0]
    u.pos = np.array([100.0, 500.0])
    f = e.checker.check(u, TaskArrays.from_infos([_info(x=900, y=500)]), e.t)
    assert f.reasons[0] & F.NOFLY


def test_infeasible_masked_even_if_actor_scores_it_highest():
    c, e = _env()
    d = feature_dims(c)
    for _ in range(10):
        e.step({})
    ids = e.decision_uavs()
    o = next(o for o in e.observe(ids) if o.n_tasks > 0)
    u = e.uavs[o.ego_id]
    u.energy = 0.3 * u.capacity                              # make far tasks energy-infeasible
    o = e.observe([o.ego_id])[0]
    bad = np.nonzero(o.task_mask & ~o.action_mask[: d.n_task])[0]
    assert len(bad) > 0
    a = GraphActor(d, c.model)
    orig = a.score_candidates

    def rigged(enc, b):
        lg = orig(enc, b)
        lg[:, bad[0]] = 1e6                                   # highest possible raw score for an infeasible task
        return lg
    a.score_candidates = rigged
    b = collate([o])
    dist, _ = a.get_action_distribution(b)
    assert dist.probs[0, bad[0]] == 0.0
    for _ in range(50):
        assert int(a.act(b)["action"]) != int(bad[0])


def test_env_overrides_infeasible_request():
    c, e = _env()
    u = e.uavs[0]
    u.energy = 0.2 * u.capacity
    tid = max(e.tasks, key=lambda t: np.linalg.norm(t.pos - u.pos)).id
    e.coord.tables[0].learn_task(TaskInfo.from_task(e.tasks[tid]), e.t)
    e.step({0: Action(ActionKind.TASK, tid)})
    assert u.task_id != tid and e.m["safety_overrides"] >= 1


def test_forced_return_when_reserve_reached_and_emergency_policy():
    c, e = _env()
    u = e.uavs[0]
    u.pos = np.array([500.0, 500.0])
    u.energy = e.checker.return_energy(u.pos) + e.checker.reserve_energy(u.capacity) - 0.1
    assert e.checker.must_return(u)
    assert 0 not in e.decision_uavs()                        # policy is not even asked
    e.step({})
    assert u.status in (UAVStatus.RETURNING, UAVStatus.FAILED)
    u2 = e.uavs[1]
    u2.pos = np.array([500.0, 500.0]); u2.energy = 0.5       # cannot make it back
    e._do_return_or_emergency(u2, e.t)
    assert u2.status == UAVStatus.FAILED and u2.fail_reason == "emergency_landing"
