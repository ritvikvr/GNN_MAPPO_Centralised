"""Typed configuration loaded from YAML, with deep-merge, CLI overrides and validation."""
from __future__ import annotations

import copy
from dataclasses import MISSING, dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class EnvConfig:
    arena: float = 1000.0
    n_uavs: int = 20
    n_tasks: int = 80
    horizon: int = 300
    dt: float = 1.0
    n_bases: int = 2
    n_capabilities: int = 3
    speed: float = 15.0
    speed_hetero: float = 0.2
    capacity: float = 40.0
    capacity_hetero: float = 0.15
    move_energy_per_m: float = 0.02
    hover_energy_per_s: float = 0.02
    service_energy_per_s: float = 0.1
    charge_rate: float = 0.5
    payload_capacity: float = 10.0
    task_payload_range: List[float] = field(default_factory=lambda: [0.0, 4.0])
    initial_task_frac: float = 0.4
    arrival_window_frac: float = 0.6
    service_range: List[float] = field(default_factory=lambda: [8.0, 25.0])
    deadline_slack_range: List[float] = field(default_factory=lambda: [60.0, 150.0])
    deadline_tightness: float = 1.0
    priority_weights: List[float] = field(default_factory=lambda: [0.3, 0.25, 0.2, 0.15, 0.1])
    hard_deadlines: bool = True
    wind_std: float = 2.0
    wind_tau: float = 60.0
    failure_rate: float = 0.0
    no_fly_zones: List[List[float]] = field(default_factory=list)
    arrival_uncertainty: float = 0.1
    redecide_period: int = 5   # steps between policy re-queries of UAVs that already hold a claim


@dataclass
class CommConfig:
    range: float = 400.0
    loss_prob: float = 0.05
    delay_min: int = 0
    delay_max: int = 2
    outage_prob: float = 0.002
    outage_min: int = 5
    outage_max: int = 20
    partition_prob: float = 0.0
    partition_duration: int = 40
    heartbeat_period: int = 2
    neighbor_timeout: int = 10
    command_loss: float = 0.05
    heartbeat_bytes: int = 48
    claim_bytes: int = 40


@dataclass
class SafetyConfig:
    reserve_frac: float = 0.15
    energy_margin: float = 0.10
    time_margin: float = 0.15
    energy_filter: bool = True
    emergency_policy: str = "land_in_place"


@dataclass
class CoordConfig:
    enabled: bool = True
    confirm_window: int = 3
    lease_duration: int = 12
    renew_period: int = 4


@dataclass
class RewardConfig:
    w_c: float = 1.0
    w_e: float = 0.5
    w_t: float = 2.0
    w_l: float = 0.5
    w_s: float = 0.1
    w_d: float = 0.5
    w_q: float = 0.002


@dataclass
class ModelConfig:
    actor_type: str = "gnn"
    aggregation: str = "attention"
    recurrent: bool = False
    hidden: int = 128
    layers: int = 2
    heads: int = 4
    max_uav_nodes: int = 6
    max_task_nodes: int = 12


@dataclass
class BCConfig:
    enabled: bool = False
    episodes: int = 10
    epochs: int = 10
    lr: float = 1e-3


@dataclass
class RandomizationConfig:
    enabled: bool = True
    fleet_sizes: List[int] = field(default_factory=lambda: [10, 15, 20])
    n_tasks_per_uav: List[float] = field(default_factory=lambda: [2.5, 4.5])
    capacity: List[float] = field(default_factory=lambda: [30.0, 50.0])
    speed: List[float] = field(default_factory=lambda: [12.0, 18.0])
    wind_std: List[float] = field(default_factory=lambda: [0.5, 3.0])
    comm_range: List[float] = field(default_factory=lambda: [250.0, 500.0])
    loss_prob: List[float] = field(default_factory=lambda: [0.0, 0.2])
    delay_max: List[int] = field(default_factory=lambda: [1, 3])
    failure_rate: List[float] = field(default_factory=lambda: [0.0, 0.0005])
    deadline_tightness: List[float] = field(default_factory=lambda: [0.7, 1.3])
    priority_skew: List[float] = field(default_factory=lambda: [0.0, 1.0])


@dataclass
class TrainConfig:
    algo: str = "mappo"
    iterations: int = 40
    episodes_per_iter: int = 6
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch_size: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_schedule: bool = False
    normalize_adv: bool = True
    critic_warmup_iters: int = 0   # MAPPO: iterations that train only the critic (actor frozen)
    eval_every: int = 5
    eval_episodes: int = 4
    device: str = "auto"
    log_dir: str = "results/tb"
    out_dir: str = "checkpoints"
    bc: BCConfig = field(default_factory=BCConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)


@dataclass
class EvalConfig:
    n_episodes: int = 10
    methods: List[str] = field(default_factory=lambda: ["nearest", "greedy", "hungarian", "local_greedy"])
    checkpoint: str = "checkpoints/best.pt"
    fleet_sizes: List[int] = field(default_factory=lambda: [10, 20, 30, 50])
    tasks_per_uav: float = 2.5
    out_dir: str = "results"
    conditions: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    coordination: CoordConfig = field(default_factory=CoordConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> "Config":
        e, c, m, t, s = self.env, self.comm, self.model, self.train, self.safety
        _req(e.arena > 0 and e.dt > 0 and e.horizon > 0, "env.arena/dt/horizon must be > 0")
        _req(1 <= e.n_uavs <= 200, "env.n_uavs must be in [1, 200]")
        _req(e.n_tasks >= 0, "env.n_tasks must be >= 0")
        _req(e.n_bases >= 1, "env.n_bases must be >= 1")
        _req(e.n_capabilities >= 1, "env.n_capabilities must be >= 1")
        _req(len(e.priority_weights) == 5 and abs(sum(e.priority_weights)) > 0, "env.priority_weights needs 5 entries")
        _req(0 <= e.initial_task_frac <= 1 and 0 < e.arrival_window_frac <= 1, "task arrival fractions out of range")
        _req(e.capacity > 0 and e.speed > 0, "env.capacity/speed must be > 0")
        _req(0 <= c.loss_prob <= 1 and 0 <= c.command_loss <= 1, "comm loss probabilities must be in [0,1]")
        _req(0 <= c.delay_min <= c.delay_max, "comm.delay_min <= delay_max required")
        _req(c.heartbeat_period >= 1, "comm.heartbeat_period must be >= 1")
        _req(0 <= s.reserve_frac < 1, "safety.reserve_frac must be in [0,1)")
        _req(s.emergency_policy in ("land_in_place", "continue_to_base"), "unknown safety.emergency_policy")
        _req(m.actor_type in ("gnn", "mlp"), "model.actor_type must be gnn|mlp")
        _req(m.aggregation in ("attention", "mean"), "model.aggregation must be attention|mean")
        _req(m.hidden % m.heads == 0, "model.hidden must be divisible by model.heads")
        _req(m.layers >= 1 and m.max_uav_nodes >= 1 and m.max_task_nodes >= 1, "model sizes must be >= 1")
        _req(t.algo in ("mappo", "ippo"), "train.algo must be mappo|ippo")
        _req(0 < t.gamma <= 1 and 0 <= t.gae_lambda <= 1 and t.clip > 0, "train gamma/lambda/clip invalid")
        _req(self.coordination.lease_duration > self.coordination.renew_period >= 1,
             "coordination.lease_duration must exceed renew_period")
        return self


def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Invalid config: {msg}")


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls, data: Dict[str, Any], path: str = ""):
    """Recursively instantiate dataclass `cls` from dict, rejecting unknown keys."""
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in (data or {}).items():
        if k not in known:
            raise KeyError(f"Unknown config key '{path}{k}' for {cls.__name__}")
        fld = known[k]
        ftype = fld.default_factory() if fld.default_factory is not MISSING else fld.default  # type: ignore
        if is_dataclass(ftype) and isinstance(v, dict):
            kwargs[k] = _build(type(ftype), v, f"{path}{k}.")
        else:
            kwargs[k] = v
    return cls(**kwargs)


def parse_override(s: str) -> Dict[str, Any]:
    """'model.hidden=64' -> {'model': {'hidden': 64}} (value parsed as YAML)."""
    key, _, val = s.partition("=")
    if not _:
        raise ValueError(f"override must look like a.b=value, got '{s}'")
    node: Dict[str, Any] = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = yaml.safe_load(val)
    return node


def load_config(*paths: Optional[str], overrides: Optional[List[str]] = None, include_default: bool = True) -> Config:
    """Merge default.yaml, then each given YAML (in order), then ``a.b=c`` overrides."""
    merged: Dict[str, Any] = {}
    files: List[Path] = []
    if include_default and DEFAULT_CONFIG.exists():
        files.append(DEFAULT_CONFIG)
    files += [Path(p) for p in paths if p]
    for f in files:
        with open(f, "r") as fh:
            merged = deep_update(merged, yaml.safe_load(fh) or {})
    for o in overrides or []:
        merged = deep_update(merged, parse_override(o))
    return _build(Config, merged).validate()


def config_from_dict(d: Dict[str, Any]) -> Config:
    return _build(Config, d).validate()


def with_changes(cfg: Config, changes: Dict[str, Dict[str, Any]]) -> Config:
    """Return a validated copy of ``cfg`` with ``{section: {key: value}}`` applied."""
    return config_from_dict(deep_update(cfg.to_dict(), changes))
