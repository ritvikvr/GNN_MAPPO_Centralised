"""Enhanced 3-D simulation viewer and video generator.
Provides step-by-step smooth trajectory rendering, per-agent tracking,
detailed task allocation logging, MP4/GIF generation, and interactive WebGL output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..environment.task import TaskStatus
from ..environment.uav import UAVStatus
from .simulator_view import TASK_COL, UAV_MARK


class AltitudeTracker:
    def __init__(self, n_uavs: int, arena_size: float = 1000.0):
        self.arena = arena_size
        self.n_uavs = n_uavs
        self.cruise_altitudes = {i: 65.0 + (i % 6) * 15.0 for i in range(n_uavs)}
        self.current_altitudes = {i: 0.0 for i in range(n_uavs)}
        self.history: List[Dict[str, Any]] = []
        self.trajectories: Dict[int, List[Tuple[float, float, float, int, float]]] = {
            i: [] for i in range(n_uavs)
        }

    def get_task_altitude(self, task) -> float:
        return 25.0 + (task.priority - 1) * 15.0

    def step(self, env) -> Dict[int, float]:
        dt = env.cfg.env.dt
        max_climb = 8.0 * dt
        max_descend = 7.0 * dt

        for u in env.uavs:
            uid = u.id
            curr_z = self.current_altitudes.get(uid, 0.0)
            z_cruise = self.cruise_altitudes.get(uid, 70.0)

            if u.status == UAVStatus.FAILED:
                target_z = 0.0
            elif u.status in (UAVStatus.IDLE, UAVStatus.CHARGING):
                d_base = min(np.linalg.norm(u.pos - b) for b in env.bases) if len(env.bases) else 0.0
                target_z = 0.0 if d_base < 25.0 else z_cruise * 0.5
            elif u.status in (UAVStatus.RESERVED, UAVStatus.EN_ROUTE):
                if 0 <= u.task_id < len(env.tasks):
                    task = env.tasks[u.task_id]
                    d_task = float(np.linalg.norm(u.pos - task.pos))
                    z_task = self.get_task_altitude(task)
                    if d_task < 60.0:
                        alpha = max(0.0, min(1.0, d_task / 60.0))
                        target_z = z_task + alpha * (z_cruise - z_task)
                    else:
                        target_z = z_cruise
                else:
                    target_z = z_cruise
            elif u.status == UAVStatus.EXECUTING:
                target_z = self.get_task_altitude(env.tasks[u.task_id]) if 0 <= u.task_id < len(env.tasks) else 35.0
            elif u.status == UAVStatus.RETURNING:
                d_base = min(np.linalg.norm(u.pos - b) for b in env.bases) if len(env.bases) else 0.0
                target_z = (d_base / 80.0) * z_cruise if d_base < 80.0 else z_cruise
            else:
                target_z = z_cruise

            if target_z > curr_z:
                new_z = min(target_z, curr_z + max_climb)
            else:
                new_z = max(target_z, curr_z - max_descend)

            self.current_altitudes[uid] = float(new_z)
            self.trajectories[uid].append((float(u.pos[0]), float(u.pos[1]), float(new_z), int(u.status), float(u.soc)))

        step_snapshot = {
            "t": int(env.t),
            "uavs": [
                {
                    "id": int(u.id),
                    "x": round(float(u.pos[0]), 2),
                    "y": round(float(u.pos[1]), 2),
                    "z": round(self.current_altitudes[u.id], 2),
                    "status": int(u.status),
                    "status_name": u.status.name,
                    "soc": round(float(u.soc), 3),
                    "task_id": int(u.task_id),
                    "energy": round(float(u.energy), 2),
                    "speed": round(float(u.speed), 1),
                }
                for u in env.uavs
            ],
            "tasks": [
                {
                    "id": int(x.id),
                    "x": round(float(x.pos[0]), 2),
                    "y": round(float(x.pos[1]), 2),
                    "z": round(self.get_task_altitude(x), 2),
                    "priority": int(x.priority),
                    "status": int(x.status),
                    "status_name": x.status.name,
                    "announced": bool(x.announced),
                    "owner": int(x.owner),
                }
                for x in env.tasks
                if x.announced
            ],
            "links": [
                [int(i), int(j)]
                for i in range(len(env.uavs))
                for j in range(i + 1, len(env.uavs))
                if env.mode != "centralized" and env.coord.adj[i, j]
            ],
            "stats": {
                "done": sum(x.status == TaskStatus.COMPLETED for x in env.tasks),
                "expired": sum(x.status == TaskStatus.EXPIRED for x in env.tasks),
                "in_progress": sum(x.status == TaskStatus.IN_PROGRESS for x in env.tasks),
                "active_links": int(np.sum(env.coord.adj) // 2) if env.mode != "centralized" else 0,
                "mean_soc": round(float(np.mean([u.soc for u in env.uavs])), 3),
            },
        }
        self.history.append(step_snapshot)
        return self.current_altitudes


def render_frame_3d(
    env,
    ax,
    tracker: AltitudeTracker,
    show_links: bool = True,
    title: str = "",
    elev: float = 30.0,
    azim: float = 45.0,
    trail_len: int = 25,
):
    ax.clear()
    A = float(env.cfg.env.arena)
    Z_MAX = 180.0

    ax.set_xlim(-20, A + 20)
    ax.set_ylim(-20, A + 20)
    ax.set_zlim(0, Z_MAX)

    ax.set_xlabel("X (m)", fontsize=8, labelpad=2)
    ax.set_ylabel("Y (m)", fontsize=8, labelpad=2)
    ax.set_zlabel("Altitude Z (m)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)

    # Base stations
    for i, b in enumerate(env.bases):
        ax.scatter([b[0]], [b[1]], [0], marker="^", s=140, c="#111111", edgecolors="#22c55e", lw=1.6, zorder=5)
        ax.text(b[0] + 15, b[1] + 15, 2, f"Base {i}", fontsize=7.5, color="#111111", weight="bold")
        theta = np.linspace(0, 2 * np.pi, 24)
        r_pad = 25.0
        ax.plot(b[0] + r_pad * np.cos(theta), b[1] + r_pad * np.sin(theta), np.zeros_like(theta), c="#16a34a", lw=1.0, alpha=0.7)

    # No-Fly Zones
    for z in env.cfg.env.no_fly_zones:
        xc, yc, r = float(z[0]), float(z[1]), float(z[2])
        theta = np.linspace(0, 2 * np.pi, 24)
        z_cyl = np.linspace(0, Z_MAX * 0.85, 8)
        Theta, Z_grid = np.meshgrid(theta, z_cyl)
        X_grid = xc + r * np.cos(Theta)
        Y_grid = yc + r * np.sin(Theta)
        ax.plot_surface(X_grid, Y_grid, Z_grid, color="#ef4444", alpha=0.15, shade=False)

    # Tasks
    for x in env.tasks:
        if not x.announced:
            continue
        z_task = tracker.get_task_altitude(x)
        c = TASK_COL.get(x.status, "#888888")
        s = 25 + 20 * x.priority

        # Drop-line and ground shadow
        ax.plot([x.pos[0], x.pos[0]], [x.pos[1], x.pos[1]], [0, z_task], ":", c=c, lw=0.6, alpha=0.45)
        ax.scatter([x.pos[0]], [x.pos[1]], [0], s=s * 0.35, c=c, alpha=0.25, marker="o")
        ax.scatter([x.pos[0]], [x.pos[1]], [z_task], s=s, c=c, edgecolors="k", linewidths=0.5, zorder=6)

        if not x.status.terminal:
            ax.text(x.pos[0] + 6, x.pos[1] + 6, z_task + 2, f"T{x.id}", fontsize=6, color="#222222")

    # UAVs
    alts = tracker.current_altitudes
    for u in env.uavs:
        z_u = alts.get(u.id, 0.0)
        col = plt.cm.RdYlGn(u.soc) if u.status != UAVStatus.FAILED else [0.1, 0.1, 0.1, 1.0]
        mark = UAV_MARK.get(u.status, "o")

        # Full or partial smooth breadcrumb trail
        traj = tracker.trajectories.get(u.id, [])
        if len(traj) > 1 and trail_len > 0:
            sub = traj[-trail_len:]
            xs = [p[0] for p in sub]
            ys = [p[1] for p in sub]
            zs = [p[2] for p in sub]
            ax.plot(xs, ys, zs, "-", c=col, lw=1.3, alpha=0.65)

        # Drop line to ground
        ax.plot([u.pos[0], u.pos[0]], [u.pos[1], u.pos[1]], [0, z_u], "--", c="#94a3b8", lw=0.6, alpha=0.4)
        ax.scatter([u.pos[0]], [u.pos[1]], [0], marker="o", s=25, c="#64748b", alpha=0.3)

        # UAV Marker
        ax.scatter([u.pos[0]], [u.pos[1]], [z_u], marker=mark, s=90, c=[col], edgecolors="k", linewidths=0.7, zorder=8)
        ax.text(u.pos[0] - 10, u.pos[1] + 6, z_u + 4, f"U{u.id}", fontsize=6.5, weight="bold")

        # Target Assignment Beam
        if u.task_id >= 0 and u.status in (UAVStatus.EN_ROUTE, UAVStatus.EXECUTING, UAVStatus.RESERVED):
            if u.task_id < len(env.tasks):
                task = env.tasks[u.task_id]
                z_t = tracker.get_task_altitude(task)
                beam_col = "#0284c7" if u.status != UAVStatus.RESERVED else "#ec4899"
                ax.plot([u.pos[0], task.pos[0]], [u.pos[1], task.pos[1]], [z_u, z_t], "--", c=beam_col, lw=1.2, alpha=0.8)

    # 3-D Communication Mesh Links
    if show_links and env.mode != "centralized":
        adj = env.coord.adj
        for i in range(len(env.uavs)):
            for j in range(i + 1, len(env.uavs)):
                if adj[i, j]:
                    p1 = [env.uavs[i].pos[0], env.uavs[i].pos[1], alts.get(i, 0.0)]
                    p2 = [env.uavs[j].pos[0], env.uavs[j].pos[1], alts.get(j, 0.0)]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c="#00b4d8", lw=0.55, alpha=0.4, zorder=4)

    done = sum(x.status == TaskStatus.COMPLETED for x in env.tasks)
    exp = sum(x.status == TaskStatus.EXPIRED for x in env.tasks)
    in_prog = sum(x.status == TaskStatus.IN_PROGRESS for x in env.tasks)
    total_announced = sum(x.announced for x in env.tasks)
    links_count = int(np.sum(env.coord.adj) // 2) if env.mode != "centralized" else 0
    mean_bat = float(np.mean([u.soc for u in env.uavs])) * 100.0

    ax.set_title(
        f"{title} | 3D Airspace Step t={env.t}s\n"
        f"Completed: {done}/{total_announced} | Expired: {exp} | In Progress: {in_prog} | RF Links: {links_count} | Mean Battery: {mean_bat:.1f}%",
        fontsize=8.5,
        pad=10,
    )
    return ax


def render_trajectories_3d(
    tracker: AltitudeTracker,
    env,
    path: str,
    title: str = "3D Multi-UAV Flight Trajectories & Airspace Allocation",
):
    fig = plt.figure(figsize=(11, 8.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    A = float(env.cfg.env.arena)
    Z_MAX = 180.0

    ax.set_xlim(-20, A + 20)
    ax.set_ylim(-20, A + 20)
    ax.set_zlim(0, Z_MAX)
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Y (m)", labelpad=4)
    ax.set_zlabel("Altitude Z (m)", labelpad=4)
    ax.view_init(elev=28, azim=50)

    # Base stations
    for i, b in enumerate(env.bases):
        ax.scatter([b[0]], [b[1]], [0], marker="^", s=180, c="#111111", edgecolors="#22c55e", lw=1.6, zorder=10)
        ax.text(b[0] + 15, b[1] + 15, 5, f"Base {i}", fontsize=9, weight="bold")

    # Tasks
    for x in env.tasks:
        if not x.announced:
            continue
        z_task = tracker.get_task_altitude(x)
        c = TASK_COL.get(x.status, "#888888")
        ax.scatter([x.pos[0]], [x.pos[1]], [z_task], s=35 + 20 * x.priority, c=c, edgecolors="k", lw=0.4, alpha=0.85)

    # Flight Paths
    cmap = matplotlib.colormaps["tab20"].resampled(max(1, len(env.uavs)))
    for uid, points in tracker.trajectories.items():
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        c = cmap(uid % 20)
        ax.plot(xs, ys, zs, "-", color=c, lw=1.6, alpha=0.85, label=f"UAV {uid}" if uid < 10 else None)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=c, s=40, marker="^")
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=c, s=50, marker="o", edgecolors="k")

    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    if len(env.uavs) <= 10:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, ncol=1)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_flight_profiles(
    tracker: AltitudeTracker,
    env,
    path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)

    ax = axes[0, 0]
    for uid, points in tracker.trajectories.items():
        times = list(range(len(points)))
        alts = [p[2] for p in points]
        ax.plot(times, alts, lw=1.2, alpha=0.75, label=f"UAV {uid}" if uid < 6 else None)
    ax.set_title("3D Altitude Profiles over Time", fontsize=10, weight="bold")
    ax.set_xlabel("Simulation Step (s)", fontsize=9)
    ax.set_ylabel("Altitude Z (m)", fontsize=9)
    ax.grid(True, alpha=0.3)
    if tracker.n_uavs <= 6:
        ax.legend(fontsize=7)

    ax = axes[0, 1]
    for uid, points in tracker.trajectories.items():
        times = list(range(len(points)))
        socs = [p[4] * 100.0 for p in points]
        ax.plot(times, socs, lw=1.2, alpha=0.75)
    ax.axhline(env.cfg.safety.reserve_frac * 100.0, color="r", linestyle="--", lw=1.0, label="Safety Reserve Limit")
    ax.set_title("UAV Battery State of Charge (%)", fontsize=10, weight="bold")
    ax.set_xlabel("Simulation Step (s)", fontsize=9)
    ax.set_ylabel("Battery SoC (%)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    steps = [h["t"] for h in tracker.history]
    dones = [h["stats"]["done"] for h in tracker.history]
    expireds = [h["stats"]["expired"] for h in tracker.history]
    in_progs = [h["stats"]["in_progress"] for h in tracker.history]
    ax.plot(steps, dones, "-g", lw=2, label="Completed Tasks")
    ax.plot(steps, expireds, "-r", lw=1.5, label="Expired Tasks")
    ax.plot(steps, in_progs, "-b", lw=1.5, label="In Progress")
    ax.set_title("Task Allocation Progress", fontsize=10, weight="bold")
    ax.set_xlabel("Simulation Step (s)", fontsize=9)
    ax.set_ylabel("Task Count", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    active_links = [h["stats"]["active_links"] for h in tracker.history]
    ax.plot(steps, active_links, color="#0077b6", lw=1.8, label="Active RF Comms Links")
    ax.set_title("Decentralized Communication Mesh Density", fontsize=10, weight="bold")
    ax.set_xlabel("Simulation Step (s)", fontsize=9)
    ax.set_ylabel("RF Links Count", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def generate_interactive_3d_html(
    tracker: AltitudeTracker,
    env,
    path: str,
    method_name: str = "GNN-MAPPO",
    task_assignment_ledger: Optional[List[Dict[str, Any]]] = None,
):
    sim_data = {
        "arena": float(env.cfg.env.arena),
        "horizon": int(env.cfg.env.horizon),
        "method": method_name,
        "bases": [[float(b[0]), float(b[1])] for b in env.bases],
        "no_fly_zones": [[float(z[0]), float(z[1]), float(z[2])] for z in env.cfg.env.no_fly_zones],
        "history": tracker.history,
        "ledger": task_assignment_ledger or [],
    }

    json_str = json.dumps(sim_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>3D UAV Fleet Mission Visualizer - {method_name}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; }}
    body {{ background: #0a0e17; color: #e2e8f0; overflow: hidden; height: 100vh; display: flex; flex-direction: column; }}
    #header {{
      background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px);
      padding: 10px 24px; display: flex; justify-content: space-between; align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1); z-index: 10;
    }}
    .title-group h1 {{ font-size: 1.15rem; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; }}
    .badge {{ background: #0284c7; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
    .stats-row {{ display: flex; gap: 14px; align-items: center; }}
    .stat-chip {{ background: rgba(30, 41, 59, 0.75); padding: 5px 12px; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.8rem; }}
    .stat-chip strong {{ color: #38bdf8; font-weight: 700; }}

    #main-container {{ position: relative; flex: 1; overflow: hidden; }}
    #viewport-3d {{ width: 100%; height: 100%; display: block; }}

    #hud-panel {{
      position: absolute; top: 14px; left: 14px; width: 280px;
      background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px;
      padding: 12px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); z-index: 5;
    }}
    #hud-panel h3 {{ font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }}
    .hud-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }}
    .hud-card {{ background: rgba(30, 41, 59, 0.65); padding: 8px; border-radius: 6px; }}
    .hud-card .val {{ font-size: 1.25rem; font-weight: 700; color: #f8fafc; }}
    .hud-card .lbl {{ font-size: 0.68rem; color: #94a3b8; margin-top: 2px; }}
    .hud-card.green .val {{ color: #4ade80; }}
    .hud-card.red .val {{ color: #f87171; }}
    .hud-card.cyan .val {{ color: #38bdf8; }}
    .hud-card.yellow .val {{ color: #facc15; }}

    #camera-tools {{ display: flex; gap: 6px; margin-top: 6px; flex-wrap: wrap; }}
    .tool-btn {{
      background: #1e293b; color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.1);
      padding: 5px 9px; border-radius: 4px; font-size: 0.72rem; cursor: pointer; transition: all 0.2s;
    }}
    .tool-btn:hover {{ background: #0284c7; color: white; border-color: #0284c7; }}

    #uav-inspector {{
      position: absolute; top: 14px; right: 14px; width: 280px;
      background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px;
      padding: 12px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); z-index: 5;
      font-size: 0.78rem;
    }}
    #uav-inspector h3 {{ font-size: 0.85rem; color: #38bdf8; margin-bottom: 6px; }}
    .prop-row {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05); }}
    .prop-label {{ color: #94a3b8; }}
    .prop-val {{ font-weight: 600; color: #f8fafc; }}

    #assignment-ledger {{
      position: absolute; bottom: 85px; right: 14px; width: 330px; max-height: 220px;
      background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px;
      padding: 10px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); z-index: 5;
      overflow-y: auto; font-size: 0.72rem;
    }}
    #assignment-ledger h3 {{ font-size: 0.8rem; color: #38bdf8; margin-bottom: 6px; }}
    table.ledger-table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    table.ledger-table th {{ color: #94a3b8; border-bottom: 1px solid #334155; padding: 3px 4px; }}
    table.ledger-table td {{ padding: 3px 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); }}
    .status-tag {{ padding: 2px 4px; border-radius: 3px; font-weight: 600; font-size: 0.65rem; }}
    .tag-completed {{ background: #166534; color: #86efac; }}
    .tag-executing {{ background: #1e3a8a; color: #93c5fd; }}
    .tag-expired {{ background: #7f1d1d; color: #fca5a5; }}

    #timeline-panel {{
      background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(14px);
      border-top: 1px solid rgba(255, 255, 255, 0.1); padding: 10px 24px;
      display: flex; align-items: center; gap: 14px; z-index: 10;
    }}
    .ctrl-btn {{
      background: #0284c7; color: white; border: none; border-radius: 6px;
      width: 36px; height: 36px; display: flex; align-items: center; justify-content: center;
      cursor: pointer; font-size: 1rem; transition: background 0.2s;
    }}
    .ctrl-btn:hover {{ background: #0369a1; }}
    #scrubber {{ flex: 1; accent-color: #38bdf8; height: 6px; border-radius: 3px; cursor: pointer; }}
    #time-display {{ font-variant-numeric: tabular-nums; font-weight: 600; font-size: 0.85rem; min-width: 90px; color: #94a3b8; }}
    #speed-select {{ background: #1e293b; color: white; border: 1px solid rgba(255, 255, 255, 0.15); padding: 4px 8px; border-radius: 4px; font-size: 0.76rem; }}

    .legend-box {{ position: absolute; bottom: 14px; left: 14px; background: rgba(15, 23, 42, 0.85); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.7rem; z-index: 5; display: flex; gap: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 5px; }}
    .legend-color {{ width: 9px; height: 9px; border-radius: 2px; }}
  </style>
</head>
<body>
  <div id="header">
    <div class="title-group">
      <h1>3D UAV Mission Visualizer <span class="badge">{method_name}</span></h1>
    </div>
    <div class="stats-row">
      <div class="stat-chip">Arena: <strong>{sim_data['arena']}m × {sim_data['arena']}m</strong></div>
      <div class="stat-chip">Fleet: <strong>{len(tracker.trajectories)} UAVs</strong></div>
      <div class="stat-chip">Tasks: <strong>{len(sim_data['ledger'])} Total</strong></div>
      <div class="stat-chip">Status: <strong style="color: #4ade80;">Playback Ready (Precise/Slow)</strong></div>
    </div>
  </div>

  <div id="main-container">
    <div id="hud-panel">
      <h3>Airspace Telemetry</h3>
      <div class="hud-grid">
        <div class="hud-card green">
          <div class="val" id="hud-done">0</div>
          <div class="lbl">Completed Tasks</div>
        </div>
        <div class="hud-card red">
          <div class="val" id="hud-expired">0</div>
          <div class="lbl">Expired Tasks</div>
        </div>
        <div class="hud-card cyan">
          <div class="val" id="hud-links">0</div>
          <div class="lbl">Mesh RF Links</div>
        </div>
        <div class="hud-card yellow">
          <div class="val" id="hud-battery">100%</div>
          <div class="lbl">Mean Battery</div>
        </div>
      </div>
      <h3>Camera Views</h3>
      <div id="camera-tools">
        <button class="tool-btn" onclick="setCamera('iso')">Isometric 3D</button>
        <button class="tool-btn" onclick="setCamera('top')">Top-Down</button>
        <button class="tool-btn" onclick="setCamera('front')">Elevation</button>
        <button class="tool-btn" onclick="trackSelectedUav()">Track Selected UAV</button>
      </div>
      <div style="margin-top: 8px;">
        <label style="font-size: 0.72rem; color: #94a3b8;">Select Agent to Track:</label>
        <select id="uav-select" onchange="selectUav(this.value)" style="width: 100%; margin-top: 3px; background: #1e293b; color: white; border: 1px solid #334155; padding: 3px 6px; border-radius: 4px; font-size: 0.74rem;">
        </select>
      </div>
    </div>

    <div id="uav-inspector">
      <h3 id="insp-title">UAV Inspector (U0)</h3>
      <div class="prop-row"><span class="prop-label">State:</span><span class="prop-val" id="insp-status">IDLE</span></div>
      <div class="prop-row"><span class="prop-label">Position (X, Y, Z):</span><span class="prop-val" id="insp-pos">0, 0, 0 m</span></div>
      <div class="prop-row"><span class="prop-label">Altitude:</span><span class="prop-val" id="insp-alt">0.0 m</span></div>
      <div class="prop-row"><span class="prop-label">Battery SoC:</span><span class="prop-val" id="insp-soc">100%</span></div>
      <div class="prop-row"><span class="prop-label">Assigned Task:</span><span class="prop-val" id="insp-task">None</span></div>
      <div class="prop-row"><span class="prop-label">Flight Speed:</span><span class="prop-val" id="insp-speed">15 m/s</span></div>
    </div>

    <div id="assignment-ledger">
      <h3>Task Assignment & Completion</h3>
      <table class="ledger-table">
        <thead>
          <tr><th>ID</th><th>Pri</th><th>Assigned</th><th>Status</th></tr>
        </thead>
        <tbody id="ledger-body">
        </tbody>
      </table>
    </div>

    <div class="legend-box">
      <div class="legend-item"><div class="legend-color" style="background:#2ca02c"></div> Executing / Done</div>
      <div class="legend-item"><div class="legend-color" style="background:#1f77b4"></div> En Route / Assigned</div>
      <div class="legend-item"><div class="legend-color" style="background:#e377c2"></div> Reserved</div>
      <div class="legend-item"><div class="legend-color" style="background:#f2a900"></div> Pending</div>
      <div class="legend-item"><div class="legend-color" style="background:#d62728"></div> Expired</div>
      <div class="legend-item"><div class="legend-color" style="background:#00b4d8"></div> RF Link</div>
    </div>

    <canvas id="viewport-3d"></canvas>
  </div>

  <div id="timeline-panel">
    <button class="ctrl-btn" id="play-btn" onclick="togglePlay()">⏸</button>
    <button class="tool-btn" onclick="stepFrame(-1)">⏮ -1</button>
    <button class="tool-btn" onclick="stepFrame(1)">⏭ +1</button>
    <input type="range" id="scrubber" min="0" max="0" value="0" oninput="onScrub(this.value)">
    <div id="time-display">t = 0s / 300s</div>
    <select id="speed-select" onchange="setSpeed(this.value)">
      <option value="0.25">0.25x (Very Slow)</option>
      <option value="0.5" selected>0.5x (Slow & Precise)</option>
      <option value="1">1.0x (Normal)</option>
      <option value="2">2.0x (Fast)</option>
    </select>
  </div>

  <script>
    const DATA = {json_str};
    const ARENA = DATA.arena;
    const historyFrames = DATA.history || [];
    let currentFrameIdx = 0;
    let isPlaying = true;
    let playSpeed = 0.5;
    let playInterval = null;
    let selectedUavId = 0;
    let isTrackingUav = false;

    // Populate select
    const uSel = document.getElementById('uav-select');
    if (historyFrames.length > 0) {{
      historyFrames[0].uavs.forEach(u => {{
        const opt = document.createElement('option');
        opt.value = u.id;
        opt.innerText = `UAV ${{u.id}} (corridor ${{Math.round(u.z)}}m)`;
        uSel.appendChild(opt);
      }});
    }}

    function selectUav(id) {{
      selectedUavId = parseInt(id);
      updateFrame(currentFrameIdx);
      if (isTrackingUav) trackSelectedUav();
    }}

    // Three.js Setup
    const canvas = document.getElementById('viewport-3d');
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d);
    scene.fog = new THREE.FogExp2(0x0a0f1d, 0.0006);

    const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 1, 10000);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    function setCamera(mode) {{
      isTrackingUav = false;
      const center = new THREE.Vector3(ARENA / 2, 0, ARENA / 2);
      controls.target.copy(center);
      if (mode === 'iso') {{
        camera.position.set(ARENA * 1.35, ARENA * 0.9, ARENA * 1.35);
      }} else if (mode === 'top') {{
        camera.position.set(ARENA / 2, ARENA * 1.5, ARENA / 2 + 1);
      }} else if (mode === 'front') {{
        camera.position.set(ARENA / 2, ARENA * 0.4, ARENA * 1.6);
      }}
      controls.update();
    }}
    setCamera('iso');

    function trackSelectedUav() {{
      isTrackingUav = true;
      const u0 = historyFrames[currentFrameIdx]?.uavs.find(u => u.id === selectedUavId);
      if (u0) {{
        controls.target.set(u0.x, u0.z, u0.y);
        camera.position.set(u0.x + 120, u0.z + 80, u0.y + 120);
        controls.update();
      }}
    }}

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(ARENA / 2, 800, ARENA / 2);
    scene.add(dirLight);

    // Ground Grid & Boundary
    const gridHelper = new THREE.GridHelper(ARENA, 40, 0x1e3a8a, 0x172554);
    gridHelper.position.set(ARENA / 2, 0, ARENA / 2);
    scene.add(gridHelper);

    const boundaryGeo = new THREE.BoxGeometry(ARENA, 2, ARENA);
    const boundaryEdges = new THREE.EdgesGeometry(boundaryGeo);
    const boundaryLines = new THREE.LineSegments(boundaryEdges, new THREE.LineBasicMaterial({{ color: 0x38bdf8, linewidth: 2 }}));
    boundaryLines.position.set(ARENA / 2, 0, ARENA / 2);
    scene.add(boundaryLines);

    // Bases
    (DATA.bases || []).forEach((b, i) => {{
      const baseGeo = new THREE.CylinderGeometry(25, 25, 3, 24);
      const baseMat = new THREE.MeshStandardMaterial({{ color: 0x10b981, roughness: 0.4 }});
      const pad = new THREE.Mesh(baseGeo, baseMat);
      pad.position.set(b[0], 1.5, b[1]);
      scene.add(pad);

      const beaconGeo = new THREE.ConeGeometry(8, 20, 16);
      const beaconMat = new THREE.MeshStandardMaterial({{ color: 0x34d399, emissive: 0x059669 }});
      const beacon = new THREE.Mesh(beaconGeo, beaconMat);
      beacon.rotation.x = Math.PI;
      beacon.position.set(b[0], 25, b[1]);
      scene.add(beacon);
    }});

    // Groups
    const uavGroup = new THREE.Group();
    const taskGroup = new THREE.Group();
    const linkGroup = new THREE.Group();
    const trailGroup = new THREE.Group();
    scene.add(uavGroup);
    scene.add(taskGroup);
    scene.add(linkGroup);
    scene.add(trailGroup);

    const STATUS_COLORS = {{
      0: 0xf2a900, 1: 0xe377c2, 2: 0x1f77b4, 3: 0x2ca02c, 4: 0x94a3b8, 5: 0xef4444, 6: 0x111111
    }};

    function updateFrame(idx) {{
      if (idx < 0 || idx >= historyFrames.length) return;
      currentFrameIdx = idx;
      const frame = historyFrames[idx];

      document.getElementById('hud-done').innerText = frame.stats.done;
      document.getElementById('hud-expired').innerText = frame.stats.expired;
      document.getElementById('hud-links').innerText = frame.stats.active_links;
      document.getElementById('hud-battery').innerText = Math.round(frame.stats.mean_soc * 100) + '%';
      document.getElementById('scrubber').value = idx;
      document.getElementById('time-display').innerText = `t = ${{frame.t}}s / ${{DATA.horizon}}s`;

      // 1. Tasks
      while(taskGroup.children.length > 0) taskGroup.remove(taskGroup.children[0]);
      (frame.tasks || []).forEach(t => {{
        const col = STATUS_COLORS[t.status] || 0x888888;
        const taskGeo = new THREE.SphereGeometry(6 + t.priority * 2, 16, 16);
        const taskMat = new THREE.MeshStandardMaterial({{ color: col, emissive: col, emissiveIntensity: 0.35 }});
        const taskMesh = new THREE.Mesh(taskGeo, taskMat);
        taskMesh.position.set(t.x, t.z, t.y);
        taskGroup.add(taskMesh);

        const lineGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(t.x, 0, t.y),
          new THREE.Vector3(t.x, t.z, t.y)
        ]);
        const lineMat = new THREE.LineBasicMaterial({{ color: col, transparent: true, opacity: 0.35 }});
        taskGroup.add(new THREE.Line(lineGeo, lineMat));
      }});

      // 2. UAVs and breadcrumb trails
      while(uavGroup.children.length > 0) uavGroup.remove(uavGroup.children[0]);
      while(trailGroup.children.length > 0) trailGroup.remove(trailGroup.children[0]);

      const uavMap = {{}};
      (frame.uavs || []).forEach(u => {{
        uavMap[u.id] = u;
        const isSel = (u.id === selectedUavId);
        const uColor = u.status === 6 ? 0x111111 : new THREE.Color().setHSL(u.soc * 0.33, 0.9, 0.5);

        const quad = new THREE.Group();
        const centerGeo = new THREE.CylinderGeometry(5, 5, 2.5, 12);
        const centerMat = new THREE.MeshStandardMaterial({{ color: uColor }});
        quad.add(new THREE.Mesh(centerGeo, centerMat));

        const armGeo = new THREE.BoxGeometry(18, 1.3, 1.3);
        const armMat = new THREE.MeshBasicMaterial({{ color: 0xcccccc }});
        const arm1 = new THREE.Mesh(armGeo, armMat);
        const arm2 = new THREE.Mesh(armGeo, armMat);
        arm2.rotation.y = Math.PI / 2;
        quad.add(arm1); quad.add(arm2);

        if (isSel) {{
          const selRing = new THREE.RingGeometry(13, 15, 24);
          const selMat = new THREE.MeshBasicMaterial({{ color: 0x38bdf8, side: THREE.DoubleSide }});
          const sMesh = new THREE.Mesh(selRing, selMat);
          sMesh.rotation.x = Math.PI / 2;
          sMesh.position.y = 1;
          quad.add(sMesh);
        }}

        quad.position.set(u.x, u.z, u.y);
        uavGroup.add(quad);

        // Drop-line
        const dropGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(u.x, 0, u.y),
          new THREE.Vector3(u.x, u.z, u.y)
        ]);
        const dropMat = new THREE.LineBasicMaterial({{ color: 0x94a3b8, transparent: true, opacity: 0.3 }});
        uavGroup.add(new THREE.Line(dropGeo, dropMat));

        // Breadcrumb Trail for this agent up to current step
        const trailPoints = [];
        for (let s = Math.max(0, idx - 40); s <= idx; s++) {{
          const uPast = historyFrames[s]?.uavs.find(pastU => pastU.id === u.id);
          if (uPast) trailPoints.push(new THREE.Vector3(uPast.x, uPast.z, uPast.y));
        }}
        if (trailPoints.length > 1) {{
          const trailGeo = new THREE.BufferGeometry().setFromPoints(trailPoints);
          const trailMat = new THREE.LineBasicMaterial({{ color: isSel ? 0x38bdf8 : uColor, transparent: true, opacity: isSel ? 0.9 : 0.5, linewidth: isSel ? 2 : 1 }});
          trailGroup.add(new THREE.Line(trailGeo, trailMat));
        }}
      }});

      // 3. Comms Links
      while(linkGroup.children.length > 0) linkGroup.remove(linkGroup.children[0]);
      (frame.links || []).forEach(pair => {{
        const u1 = uavMap[pair[0]];
        const u2 = uavMap[pair[1]];
        if (u1 && u2) {{
          const lGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(u1.x, u1.z, u1.y),
            new THREE.Vector3(u2.x, u2.z, u2.y)
          ]);
          const lMat = new THREE.LineBasicMaterial({{ color: 0x00b4d8, transparent: true, opacity: 0.4 }});
          linkGroup.add(new THREE.Line(lGeo, lMat));
        }}
      }});

      // Update Inspector
      const selUav = uavMap[selectedUavId];
      if (selUav) {{
        document.getElementById('insp-title').innerText = `UAV Inspector (U${{selUav.id}})`;
        document.getElementById('insp-status').innerText = selUav.status_name;
        document.getElementById('insp-pos').innerText = `${{selUav.x}}, ${{selUav.y}}, ${{selUav.z}} m`;
        document.getElementById('insp-alt').innerText = `${{selUav.z}} m`;
        document.getElementById('insp-soc').innerText = `${{Math.round(selUav.soc * 100)}}% (${{selUav.energy}} J)`;
        document.getElementById('insp-task').innerText = selUav.task_id >= 0 ? `Task T${{selUav.task_id}}` : 'None (Idle/Patrol)';
        document.getElementById('insp-speed').innerText = `${{selUav.speed}} m/s`;

        if (isTrackingUav) {{
          controls.target.set(selUav.x, selUav.z, selUav.y);
        }}
      }}

      // Update Ledger
      const tbody = document.getElementById('ledger-body');
      tbody.innerHTML = '';
      (DATA.ledger || []).slice(0, 15).forEach(row => {{
        const tr = document.createElement('tr');
        const isCurTask = selUav && selUav.task_id === row.task_id;
        if (isCurTask) tr.style.background = 'rgba(56, 189, 248, 0.2)';
        tr.innerHTML = `
          <td><strong>T${{row.task_id}}</strong></td>
          <td>P${{row.priority}}</td>
          <td>${{row.agent >= 0 ? 'U' + row.agent : 'None'}}</td>
          <td><span class="status-tag ${{row.status === 'COMPLETED' ? 'tag-completed' : (row.status === 'EXPIRED' ? 'tag-expired' : 'tag-executing')}}">${{row.status}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    function startPlayback() {{
      if (playInterval) clearInterval(playInterval);
      // Slower, smooth step interval
      const ms = Math.max(80, Math.round(200 / playSpeed));
      playInterval = setInterval(() => {{
        if (currentFrameIdx < historyFrames.length - 1) {{
          updateFrame(currentFrameIdx + 1);
        }} else {{
          updateFrame(0);
        }}
      }}, ms);
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById('play-btn').innerText = isPlaying ? '⏸' : '▶';
      if (isPlaying) startPlayback();
      else if (playInterval) clearInterval(playInterval);
    }}

    function onScrub(val) {{
      if (isPlaying) togglePlay();
      updateFrame(parseInt(val));
    }}

    function stepFrame(delta) {{
      if (isPlaying) togglePlay();
      updateFrame(Math.max(0, Math.min(historyFrames.length - 1, currentFrameIdx + delta)));
    }}

    function setSpeed(s) {{
      playSpeed = parseFloat(s);
      if (isPlaying) startPlayback();
    }}

    window.addEventListener('resize', () => {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }});

    if (historyFrames.length > 0) {{
      document.getElementById('scrubber').max = historyFrames.length - 1;
      updateFrame(0);
      startPlayback();
    }}
  </script>
</body>
</html>
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
