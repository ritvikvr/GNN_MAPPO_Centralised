"""2-D simulation viewer: UAVs, tasks, assignments, links, batteries, failures, reservation conflicts."""
from __future__ import annotations

from typing import Callable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..environment.task import TaskStatus
from ..environment.uav import UAVStatus

TASK_COL = {TaskStatus.PENDING: "#f2a900", TaskStatus.RESERVED: "#e377c2", TaskStatus.ASSIGNED: "#1f77b4",
            TaskStatus.IN_PROGRESS: "#2ca02c", TaskStatus.COMPLETED: "#bbbbbb", TaskStatus.EXPIRED: "#d62728", TaskStatus.FAILED: "k"}
UAV_MARK = {UAVStatus.IDLE: "o", UAVStatus.RESERVED: "s", UAVStatus.EN_ROUTE: ">", UAVStatus.EXECUTING: "*",
            UAVStatus.RETURNING: "<", UAVStatus.CHARGING: "P", UAVStatus.FAILED: "X"}


def render_frame(env, ax=None, show_links: bool = True, title: str = ""):
    """Draw the CURRENT state of ``env`` (ground truth, for humans - not what the UAVs know)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.clear()
    A = env.cfg.env.arena
    ax.set_xlim(-20, A + 20)
    ax.set_ylim(-20, A + 20)
    ax.set_aspect("equal")
    for z in env.cfg.env.no_fly_zones:
        ax.add_patch(plt.Circle(z[:2], z[2], color="red", alpha=0.15))
    ax.scatter(env.bases[:, 0], env.bases[:, 1], marker="^", s=150, c="k", label="base")
    if show_links and env.mode != "centralized":
        adj = env.coord.adj
        for i in range(len(env.uavs)):
            for j in range(i + 1, len(env.uavs)):
                if adj[i, j]:
                    ax.plot(*zip(env.uavs[i].pos, env.uavs[j].pos), c="#cccccc", lw=0.4, zorder=0)
    for x in env.tasks:
        if not x.announced:
            continue
        ax.scatter(*x.pos, s=20 + 25 * x.priority, c=TASK_COL[x.status], edgecolors="k", linewidths=0.4, zorder=2)
        if not x.status.terminal:
            ax.annotate(f"T{x.id}", x.pos, fontsize=6, xytext=(3, 3), textcoords="offset points")
    for u in env.uavs:
        col = plt.cm.RdYlGn(u.soc)
        ax.scatter(*u.pos, marker=UAV_MARK[u.status], s=70, c=[col] if u.status != UAVStatus.FAILED else "k", edgecolors="k", zorder=3)
        ax.annotate(f"{u.id}", u.pos, fontsize=6, xytext=(-8, 4), textcoords="offset points")
        if u.task_id >= 0 and u.status in (UAVStatus.EN_ROUTE, UAVStatus.EXECUTING, UAVStatus.RESERVED):
            ax.plot(*zip(u.pos, env.tasks[u.task_id].pos), "--", c="#1f77b4" if u.status != UAVStatus.RESERVED else "#e377c2", lw=0.8)
    for (tt, tid, loser, win) in env.coord.metrics.get("conflict_log", []) if env.mode != "centralized" else []:
        if env.t - tt <= 5:       # recent reservation conflict: ring around the contested task
            ax.add_patch(plt.Circle(env.tasks[tid].pos, 25, fill=False, ec="red", lw=1.2))
    done = sum(x.status == TaskStatus.COMPLETED for x in env.tasks)
    exp = sum(x.status == TaskStatus.EXPIRED for x in env.tasks)
    ax.set_title(f"{title} t={env.t}  done={done} expired={exp}  (colour=battery, ring=recent reservation conflict)", fontsize=8)
    return ax


def animate(env_factory: Callable[[], object], runner: Callable, path: str, every: int = 4, fps: int = 6) -> str:
    """Runs one episode via ``runner(env, frames_list)`` collecting snapshots, then writes a GIF/PNG-sequence."""
    from PIL import Image
    import io

    env = env_factory()
    frames: List[Image.Image] = []
    fig, ax = plt.subplots(figsize=(6, 6))

    def hook(e):
        buf = io.BytesIO()
        render_frame(e, ax)
        fig.savefig(buf, format="png", dpi=70)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P"))

    runner(env, hook, every)
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return path
