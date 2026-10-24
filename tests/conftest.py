import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from uav_allocation.config import load_config, with_changes  # noqa: E402


@pytest.fixture()
def cfg():
    c = load_config()
    return with_changes(c, {"env": {"n_uavs": 10, "n_tasks": 24, "horizon": 150}})


def small_cfg(**env):
    return with_changes(load_config(), {"env": {"n_uavs": 10, "n_tasks": 24, "horizon": 150, **env}})
