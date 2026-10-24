import inspect
from dataclasses import replace

import torch

from uav_allocation.config import load_config, with_changes
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.graph.features import feature_dims
from uav_allocation.graph.observations import collate
from uav_allocation.models.graph_actor import GraphActor
from uav_allocation.models import graph_actor


def _batch(n=6):
    c = with_changes(load_config(), {"env": {"n_uavs": 10, "n_tasks": 24, "horizon": 150}})
    e = UAVTaskEnv(c)
    e.reset(seed=1)
    for _ in range(15):
        e.step({})
    ids = e.decision_uavs()[:n]
    return c, e, e.observe(ids)


def test_variants_forward_and_masking():
    c, e, obs = _batch()
    d = feature_dims(c)
    b = collate(obs)
    for kw in [{}, {"actor_type": "mlp"}, {"aggregation": "mean"}, {"recurrent": True}]:
        a = GraphActor(d, replace(c.model, **kw))
        dist, enc = a.get_action_distribution(b)
        assert dist.probs.shape == (len(obs), d.n_action)
        assert torch.all(dist.probs[~b["action_mask"]] == 0)
        out = a.act(b, deterministic=False)
        assert b["action_mask"][torch.arange(len(obs)), out["action"]].all()
        assert out["log_prob"].shape == (len(obs),)


def test_deterministic_is_argmax_and_stable():
    c, e, obs = _batch()
    a = GraphActor(feature_dims(c), c.model)
    b = collate(obs)
    x, y = a.act(b, deterministic=True)["action"], a.act(b, deterministic=True)["action"]
    assert torch.equal(x, y)


def test_single_feasible_action_and_no_candidates():
    c, e, obs = _batch(2)
    d = feature_dims(c)
    o = obs[0]
    o.action_mask[:] = False
    o.action_mask[d.n_task] = True                     # only WAIT
    o.task_mask[:] = False                             # no tasks at all
    o.adj_ut[:] = False
    a = GraphActor(d, c.model)
    dist, _ = a.get_action_distribution(collate([o]))
    assert dist.probs[0, d.n_task] == 1.0 and abs(float(dist.entropy().detach())) < 1e-6
    o.action_mask[:] = False                           # even a fully empty mask must stay legal (hold)
    dist, _ = a.get_action_distribution(collate([o]))
    assert dist.probs[0, d.n_task] == 1.0


def test_actor_does_not_depend_on_critic_or_global_state():
    src = inspect.getsource(graph_actor)
    assert "centralized_critic" not in src and "g_uav" not in src
    c, e, obs = _batch(2)
    b = collate(obs)
    assert not any(k.startswith("g_") for k in b)       # act() works with local keys only
    GraphActor(feature_dims(c), c.model).act(b)


def test_recurrent_state_changes_output():
    c, e, obs = _batch(2)
    a = GraphActor(feature_dims(c), replace(c.model, recurrent=True))
    b = collate(obs)
    o1 = a.act(b, None)
    o2 = a.act(b, o1["hidden"])
    assert not torch.allclose(o1["hidden"], o2["hidden"])
