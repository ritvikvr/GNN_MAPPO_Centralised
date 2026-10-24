from uav_allocation.algorithms.rollout import run_episode
from uav_allocation.baselines import CENTRALIZED
from uav_allocation.config import load_config, with_changes
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.graph.features import feature_dims
from uav_allocation.graph.observations import LocalGreedyPolicy
from uav_allocation.utils import split_of, test_seed as _test_seed, train_seed, val_seed


def _cfg():
    return with_changes(load_config(), {"env": {"n_uavs": 8, "n_tasks": 16, "horizon": 120}})


def test_same_seed_same_result_centralized_and_decentralized():
    c = _cfg()
    a = run_episode(UAVTaskEnv(c), CENTRALIZED["hungarian"](), 11)
    b = run_episode(UAVTaskEnv(c), CENTRALIZED["hungarian"](), 11)
    assert {k: v for k, v in a.items() if k != "wall_s"} == {k: v for k, v in b.items() if k != "wall_s"} or all(
        (a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]) for k in a if k != "wall_s")
    p = LocalGreedyPolicy(c, feature_dims(c))
    x = run_episode(UAVTaskEnv(c), p, 12)
    y = run_episode(UAVTaskEnv(c), p, 12)
    assert x["episode_return"] == y["episode_return"] and x["energy"] == y["energy"]


def test_different_seed_differs():
    c = _cfg()
    a = run_episode(UAVTaskEnv(c), CENTRALIZED["nearest"](), 1)
    b = run_episode(UAVTaskEnv(c), CENTRALIZED["nearest"](), 2)
    assert a["energy"] != b["energy"]


def test_seed_partitions_never_overlap():
    assert split_of(train_seed(999_999)) == "train" and split_of(val_seed(5)) == "val" and split_of(_test_seed(0)) == "test"
