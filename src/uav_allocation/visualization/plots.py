"""Plots generated from CSV data of real runs only. No synthetic data."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def baseline_bars(agg: pd.DataFrame, path, metric: str, group: str = "condition") -> None:
    groups = list(agg[group].unique())
    methods = list(agg["method"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(methods) * len(groups) / 1.5 + 3, 4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        d = agg[agg["method"] == m].set_index(group).reindex(groups)
        ax.bar(np.arange(len(groups)) + i * w, d[metric + "_mean"], w, yerr=d[metric + "_ci95"], capsize=3, label=m)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - w / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel(metric + " (mean, 95% CI)")
    ax.legend(fontsize=8)
    ax.set_title(f"{metric} by method (identical seeds)")
    _save(fig, path)


def robustness(agg: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["weighted_completion_rate", "duplicate_rate"]):
        for m in agg["method"].unique():
            d = agg[agg["method"] == m]
            ax.errorbar(d["condition"], d[metric + "_mean"], yerr=d[metric + "_ci95"], marker="o", capsize=3, label=m)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)
    fig.suptitle("Robustness to communication / UAV failure conditions")
    _save(fig, path)


def scalability(agg: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["weighted_completion_rate", "msgs_per_uav_min"]):
        for m in agg["method"].unique():
            d = agg[agg["method"] == m].sort_values("fleet_size")
            ax.errorbar(d["fleet_size"], d[metric + "_mean"], yerr=d[metric + "_ci95"], marker="o", capsize=3, label=m)
        ax.set_xlabel("fleet size")
        ax.set_ylabel(metric)
    axes[0].legend(fontsize=8)
    _save(fig, path)


def training_curves(csv_path, path) -> None:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["iteration"], df["train_episode_return"], label="train (stochastic)")
    if "val_episode_return" in df:
        v = df.dropna(subset=["val_episode_return"])
        axes[0].plot(v["iteration"], v["val_episode_return"], "o-", label="validation (deterministic)")
    axes[0].set_ylabel("episode return")
    axes[0].legend()
    axes[1].plot(df["iteration"], df["train_weighted_completion_rate"], label="train")
    if "val_weighted_completion_rate" in df:
        axes[1].plot(v["iteration"], v["val_weighted_completion_rate"], "o-", label="validation")
    axes[1].set_ylabel("weighted completion rate")
    for a in axes:
        a.set_xlabel("iteration")
    _save(fig, path)


def completion_vs_time(histories: dict, path) -> None:
    """histories: {method: completed_value list per step} from real runs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, h in histories.items():
        ax.plot(h, label=m)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative completed value")
    ax.legend()
    _save(fig, path)
