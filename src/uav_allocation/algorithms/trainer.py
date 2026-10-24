"""Training driver: domain randomization, rollout collection over many simulator instances, PPO updates,
validation on held-out seeds, checkpointing (best / last), TensorBoard + CSV logging, resume."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import Config, with_changes
from ..environment.simulator import UAVTaskEnv
from ..graph.features import feature_dims
from ..utils import get_device, get_logger, seed_everything, split_of, train_seed, val_seed
from .behavior_cloning import bc_train, collect_demos
from .mappo import MAPPO
from .rollout import ActorPolicy, run_episode
from .rollout_buffer import EpisodeData, RolloutBuffer

log = get_logger("train")


def randomize_config(cfg: Config, rng: np.random.Generator) -> Config:
    """Sample one scenario from the domain-randomization ranges (all ranges live in the config)."""
    r = cfg.train.randomization
    if not r.enabled:
        return cfg
    u = lambda lohi: float(rng.uniform(lohi[0], lohi[1]))  # noqa: E731
    n_uav = int(rng.choice(r.fleet_sizes))
    n_task = int(round(n_uav * u(r.n_tasks_per_uav)))
    skew = u(r.priority_skew)
    base_w = np.asarray(cfg.env.priority_weights, dtype=float)
    heavy = base_w[::-1]
    w = ((1 - skew) * base_w / base_w.sum() + skew * heavy / heavy.sum()).tolist()
    return with_changes(cfg, {
        "env": {"n_uavs": n_uav, "n_tasks": min(n_task, 128), "capacity": u(r.capacity), "speed": u(r.speed), "wind_std": u(r.wind_std),
                "failure_rate": u(r.failure_rate), "deadline_tightness": u(r.deadline_tightness), "priority_weights": w},
        "comm": {"range": u(r.comm_range), "loss_prob": u(r.loss_prob),
                 "delay_max": int(rng.integers(r.delay_max[0], r.delay_max[1] + 1))},
    })


class Trainer:
    def __init__(self, cfg: Config, seed: int = 0, resume: Optional[str] = None, tag: str = "run"):
        self.cfg, self.seed, self.tag = cfg, seed, tag
        seed_everything(seed)
        self.device = get_device(cfg.train.device)
        self.rng = np.random.default_rng(seed)
        self.algo = MAPPO.load(resume, self.device, cfg) if resume else MAPPO(cfg, self.device)
        self.dims = feature_dims(cfg)
        self.out = Path(cfg.train.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        self.csv_path = Path("results") / f"train_log_{tag}.csv"
        self.ep_counter = self.algo.iteration * cfg.train.episodes_per_iter
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(Path(cfg.train.log_dir) / tag))
        except Exception as e:  # tensorboard optional at runtime
            log.warning("TensorBoard unavailable (%s); logging to CSV only", e)
        self._csv_header: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def behavior_cloning(self) -> Dict[str, float]:
        bc = self.cfg.train.bc
        seeds = [train_seed(900_000 + i) for i in range(bc.episodes)]
        rng = np.random.default_rng(self.seed + 7)
        cfgs = {s: randomize_config(self.cfg, rng) for s in seeds}
        t0 = time.time()
        obs, acts = collect_demos(lambda s: cfgs[s], seeds)
        stats = bc_train(self.algo.actor, obs, acts, self.device, bc.epochs, bc.lr)
        log.info("BC: %d samples, loss %.3f acc %.3f (%.0fs)", len(obs), stats["bc_loss"], stats["bc_acc"], time.time() - t0)
        return stats

    def collect(self) -> (RolloutBuffer, List[Dict[str, float]]):
        buf, sums = RolloutBuffer(), []
        pol = ActorPolicy(self.algo.actor, self.dims, self.device, deterministic=False)
        for _ in range(self.cfg.train.episodes_per_iter):
            s = train_seed(self.ep_counter)
            assert split_of(s) == "train"
            self.ep_counter += 1
            cfg_ep = randomize_config(self.cfg, np.random.default_rng(self.seed * 1_000_003 + s))
            env = UAVTaskEnv(cfg_ep)
            ep = EpisodeData()
            sums.append(run_episode(env, pol, s, record=ep))
            buf.add(ep)
        return buf, sums

    @torch.no_grad()
    def validate(self, n: int) -> Dict[str, float]:
        pol = ActorPolicy(self.algo.actor, self.dims, self.device, deterministic=True)
        rows = []
        for i in range(n):
            s = val_seed(i)
            assert split_of(s) == "val"
            rows.append(run_episode(UAVTaskEnv(self.cfg), pol, s))
        return {k: float(np.nanmean([r[k] for r in rows])) for k in ("episode_return", "weighted_completion_rate", "energy", "duplicate_rate")}

    # ------------------------------------------------------------------
    def run(self) -> Path:
        tc = self.cfg.train
        if tc.bc.enabled and self.algo.iteration == 0:
            self.behavior_cloning()
        t_start = time.time()
        start_it = self.algo.iteration
        for it in range(start_it, tc.iterations):
            t0 = time.time()
            buf, sums = self.collect()
            t_collect = time.time() - t0
            st = self.algo.update(buf)
            self.algo.iteration = it + 1
            row = {"iteration": it + 1, "episodes": self.ep_counter, "wall_s": time.time() - t_start, "collect_s": t_collect, **st}
            for k in ("episode_return", "weighted_completion_rate", "total_completion_rate", "energy", "duplicate_rate", "deadline_violations"):
                row["train_" + k] = float(np.nanmean([s[k] for s in sums]))
            if (it + 1) % tc.eval_every == 0 or it + 1 == tc.iterations:
                v = self.validate(tc.eval_episodes)
                row.update({"val_" + k: x for k, x in v.items()})
                if v["episode_return"] > self.algo.best_metric:
                    self.algo.best_metric = v["episode_return"]
                    self.algo.save(str(self.out / f"best_{self.tag}.pt"), {"val": v})
                    self.algo.save(str(self.out / "best.pt"), {"val": v})
            self.algo.save(str(self.out / f"last_{self.tag}.pt"))
            self._log(row)
        return self.out / f"best_{self.tag}.pt"

    def _log(self, row: Dict[str, float]) -> None:
        log.info("it %3d | train_ret %7.2f | wcr %.3f | dup %.3f | pi %.4f v %.4f ent %.3f kl %.4f | collect %.0fs upd %.0fs%s",
                 row["iteration"], row["train_episode_return"], row["train_weighted_completion_rate"], row["train_duplicate_rate"],
                 row.get("pi_loss", 0), row.get("v_loss", 0), row.get("entropy", 0), row.get("kl", 0), row["collect_s"], row.get("update_s", 0),
                 (" | VAL ret %.2f wcr %.3f" % (row["val_episode_return"], row["val_weighted_completion_rate"])) if "val_episode_return" in row else "")
        if self.tb is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    self.tb.add_scalar(k, v, row["iteration"])
            self.tb.flush()
        import pandas as pd

        if not hasattr(self, "_rows"):
            self._rows = pd.read_csv(self.csv_path).to_dict("records") if self.csv_path.exists() and self.algo.iteration > 1 else []
        self._rows = [r for r in self._rows if r.get("iteration") != row["iteration"]] + [row]
        pd.DataFrame(self._rows).to_csv(self.csv_path, index=False)
