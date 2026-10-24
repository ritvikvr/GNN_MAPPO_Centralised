import numpy as np
import torch

from uav_allocation.algorithms.advantages import compute_gae, discounted_returns
from uav_allocation.algorithms.mappo import MAPPO, load_actor
from uav_allocation.algorithms.rollout import ActorPolicy, run_episode
from uav_allocation.algorithms.rollout_buffer import EpisodeData, RolloutBuffer
from uav_allocation.algorithms.behavior_cloning import bc_train, collect_demos
from uav_allocation.config import load_config, with_changes
from uav_allocation.environment.simulator import UAVTaskEnv


def _cfg(**train):
    return with_changes(load_config(), {"env": {"n_uavs": 6, "n_tasks": 12, "horizon": 80}, "train": {"epochs": 1, "minibatch_size": 64, **train},
                                        "model": {"hidden": 32, "heads": 4}})


def test_gae_matches_manual_computation():
    r = np.array([1.0, 0.0, 2.0]); v = np.array([0.5, 0.4, 0.3])
    adv, ret = compute_gae(r, v, last_value=0.0, gamma=0.9, lam=0.8)
    d2 = 2.0 - 0.3; d1 = 0.0 + 0.9 * 0.3 - 0.4; d0 = 1.0 + 0.9 * 0.4 - 0.5
    a2 = d2; a1 = d1 + 0.9 * 0.8 * a2; a0 = d0 + 0.9 * 0.8 * a1
    assert np.allclose(adv, [a0, a1, a2], atol=1e-5) and np.allclose(ret, adv + v, atol=1e-5)
    adv_t, _ = compute_gae(r, v, last_value=5.0, gamma=0.9, lam=0.8)       # truncation is bootstrapped
    assert adv_t[-1] > adv[-1]
    assert np.allclose(discounted_returns(np.array([1.0, 1.0]), 0.0, 0.5), [1.5, 1.0])


def _collect(algo, cfg, n=2):
    buf = RolloutBuffer()
    pol = ActorPolicy(algo.actor, algo.dims, algo.device)
    for s in range(n):
        ep = EpisodeData()
        run_episode(UAVTaskEnv(cfg), pol, s, record=ep)
        assert len(ep.g_states) == len(ep.rewards)
        buf.add(ep)
    return buf


def test_update_changes_weights_and_is_finite():
    cfg = _cfg()
    algo = MAPPO(cfg, torch.device("cpu"))
    before = [p.detach().clone() for p in algo.actor.parameters()]
    cb = [p.detach().clone() for p in algo.critic.parameters()]
    st = algo.update(_collect(algo, cfg))
    assert all(np.isfinite(v) for v in st.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(cb, algo.critic.parameters()))


def test_ippo_and_recurrent_variants_update():
    for kw in [dict(algo="ippo"), dict()]:
        cfg = _cfg(**kw)
        if not kw:
            cfg = with_changes(cfg, {"model": {"recurrent": True}})
        algo = MAPPO(cfg, torch.device("cpu"))
        assert (algo.critic is None) == (kw.get("algo") == "ippo")
        st = algo.update(_collect(algo, cfg, 1))
        assert np.isfinite(st["pi_loss"])


def test_checkpoint_roundtrip_and_actor_only_load(tmp_path):
    cfg = _cfg()
    algo = MAPPO(cfg, torch.device("cpu"))
    algo.update(_collect(algo, cfg, 1)); algo.iteration = 3
    p = str(tmp_path / "c.pt")
    algo.save(p)
    re = MAPPO.load(p, torch.device("cpu"))
    assert re.iteration == 3
    for a, b in zip(algo.actor.parameters(), re.actor.parameters()):
        assert torch.equal(a, b)
    actor, c2, dims = load_actor(p, torch.device("cpu"))
    m1 = run_episode(UAVTaskEnv(cfg), ActorPolicy(algo.actor, algo.dims, torch.device("cpu"), deterministic=True), 7)
    m2 = run_episode(UAVTaskEnv(cfg), ActorPolicy(actor, dims, torch.device("cpu"), deterministic=True), 7)
    assert m1["episode_return"] == m2["episode_return"]
    ck = torch.load(p, weights_only=False)
    assert {"actor_opt", "config", "norm"} <= set(ck)


def test_behavior_cloning_reduces_loss():
    cfg = _cfg()
    algo = MAPPO(cfg, torch.device("cpu"))
    obs, acts = collect_demos(lambda s: cfg, [1, 2])
    s1 = bc_train(algo.actor, obs, acts, torch.device("cpu"), epochs=1)
    s2 = bc_train(algo.actor, obs, acts, torch.device("cpu"), epochs=8)
    assert s2["bc_loss"] < s1["bc_loss"]
