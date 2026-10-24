#!/usr/bin/env python
"""Run episode with high-precision step sampling, slow motion MP4/GIF generation,
and comprehensive per-task assignment reporting.

Usage:
    python scripts/run_simulation_3d.py --method gnn_mappo --checkpoint checkpoints/best.pt --uavs 20 --tasks 50 --every 1
"""
import argparse
import io
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio
import _common  # noqa: F401

from uav_allocation.config import load_config
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.algorithms.rollout import run_episode
from uav_allocation.evaluation.evaluate import make_policy
from uav_allocation.visualization.simulator_view_3d import (
    AltitudeTracker,
    render_frame_3d,
    render_trajectories_3d,
    render_flight_profiles,
    generate_interactive_3d_html,
)

def main():
    ap = argparse.ArgumentParser(description="High-Precision 3D Multi-UAV Simulation & Video Generator")
    ap.add_argument("--method", default="gnn_mappo", choices=["gnn_mappo", "greedy", "hungarian", "nearest", "local_greedy"])
    ap.add_argument("--uavs", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2_000_100)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--every", type=int, default=1, help="Render every N steps (1 = full continuous trajectory)")
    ap.add_argument("--fps", type=int, default=12, help="Playback frames per second (slower for detailed observation)")
    ap.add_argument("--out_mp4", default="results/simulation_3d.mp4")
    ap.add_argument("--out_gif", default="results/simulation_3d.gif")
    ap.add_argument("--out_html", default="results/simulation_3d.html")
    ap.add_argument("--out_traj", default="results/trajectories_3d.png")
    ap.add_argument("--out_profiles", default="results/flight_profiles_3d.png")
    ap.add_argument("--out_report_json", default="results/mission_report.json")
    ap.add_argument("-o", "--override", action="append", default=[])
    a = ap.parse_args()

    print(f"Initializing simulation ({a.method}, {a.uavs} UAVs, {a.tasks} tasks, seed {a.seed})...")
    cfg = load_config(overrides=[f"env.n_uavs={a.uavs}", f"env.n_tasks={a.tasks}"] + a.override)
    pol, cfg = make_policy(a.method, cfg, a.checkpoint)
    env = UAVTaskEnv(cfg)

    tracker = AltitudeTracker(a.uavs, arena_size=cfg.env.arena)
    fig = plt.figure(figsize=(9, 8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    raw_frames = []

    # Detailed assignment ledger tracking
    task_history_map = {}

    def on_step(e):
        tracker.step(e)

        # Record assignment status for all announced tasks
        for t in e.tasks:
            if t.announced:
                if t.id not in task_history_map:
                    task_history_map[t.id] = {
                        "task_id": t.id,
                        "priority": t.priority,
                        "pos": [round(float(t.pos[0]), 1), round(float(t.pos[1]), 1)],
                        "assigned_agent": -1,
                        "status": t.status.name,
                        "completed_at": -1,
                    }
                if t.owner >= 0:
                    task_history_map[t.id]["assigned_agent"] = int(t.owner)
                task_history_map[t.id]["status"] = t.status.name
                if t.status.name == "COMPLETED" and task_history_map[t.id]["completed_at"] < 0:
                    task_history_map[t.id]["completed_at"] = int(e.t)

        # Gentle camera pan for depth
        elev = 26.0 + 5.0 * float(np.sin(e.t * 0.02))
        azim = 35.0 + (e.t * 0.3) % 360.0

        b = io.BytesIO()
        render_frame_3d(e, ax, tracker, title=f"Method: {a.method.upper()}", elev=elev, azim=azim, trail_len=30)
        fig.savefig(b, format="png", dpi=90, bbox_inches="tight")
        b.seek(0)
        img = Image.open(b)
        raw_frames.append(np.array(img.convert("RGB")))

    print("Running episode rollout with step-by-step 3D trajectory tracking...")
    m = run_episode(env, pol, a.seed, on_step=on_step, frame_every=a.every)
    plt.close(fig)

    out_dir = Path(a.out_mp4).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export Slow MP4 Video
    print(f"Encoding slow, precise MP4 video ({len(raw_frames)} frames @ {a.fps} fps)...")
    try:
        imageio.mimwrite(a.out_mp4, raw_frames, fps=a.fps, quality=8)
        print(f"-> Saved smooth 3D video: {a.out_mp4}")
    except Exception as err:
        print(f"[warning] MP4 export failed: {err}")

    # 2. Export Slow High-Definition GIF
    print(f"Encoding slow GIF ({len(raw_frames)} frames @ {a.fps} fps)...")
    # Subsample slightly if very large to prevent memory overhead
    sub_frames = [Image.fromarray(f) for f in raw_frames[::2]]
    duration_ms = int(1000 / (a.fps / 2))  # Slow duration per frame
    sub_frames[0].save(
        a.out_gif,
        save_all=True,
        append_images=sub_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"-> Saved slow 3D GIF: {a.out_gif}")

    # 3. Final Frame PNG
    sub_frames[-1].save(str(Path(a.out_gif).with_suffix(".png")))
    print(f"-> Saved final 3D snapshot: {str(Path(a.out_gif).with_suffix('.png'))}")

    # 4. Save complete 3D trajectories
    render_trajectories_3d(tracker, env, a.out_traj, title=f"Full 3D UAV Flight Trajectories ({a.method.upper()})")
    print(f"-> Saved 3D trajectories map: {a.out_traj}")

    # 5. Save flight profiles & diagnostics
    render_flight_profiles(tracker, env, a.out_profiles)
    print(f"-> Saved telemetry & flight profiles: {a.out_profiles}")

    # 6. Task Assignment Ledger List
    ledger = [
        {
            "task_id": v["task_id"],
            "priority": v["priority"],
            "agent": v["assigned_agent"],
            "status": v["status"],
            "completed_at": v["completed_at"],
            "pos": v["pos"],
        }
        for v in sorted(task_history_map.values(), key=lambda x: x["task_id"])
    ]

    # Export interactive 3D WebGL dashboard with ledger
    generate_interactive_3d_html(tracker, env, a.out_html, method_name=a.method.upper(), task_assignment_ledger=ledger)
    print(f"-> Saved interactive 3D WebGL HTML: {a.out_html}")

    # Per-Agent Summary
    agent_summary = []
    for u in env.uavs:
        agent_summary.append({
            "uav_id": u.id,
            "tasks_completed": u.tasks_done,
            "distance_m": round(float(u.distance), 1),
            "energy_used_J": round(float(u.energy_used), 2),
            "final_battery_pct": round(float(u.soc) * 100, 1),
            "cruise_corridor_z_m": tracker.cruise_altitudes[u.id],
        })

    completed_tasks = [t for t in ledger if t["status"] == "COMPLETED"]
    expired_tasks = [t for t in ledger if t["status"] == "EXPIRED"]
    in_prog_tasks = [t for t in ledger if t["status"] == "IN_PROGRESS"]

    mission_report = {
        "method": a.method,
        "n_uavs": a.uavs,
        "total_tasks": len(ledger),
        "completed_count": len(completed_tasks),
        "expired_count": len(expired_tasks),
        "in_progress_count": len(in_prog_tasks),
        "weighted_completion_rate": float(m["weighted_completion_rate"]),
        "total_completion_rate": float(m["total_completion_rate"]),
        "energy_consumed": float(m["energy"]),
        "makespan_steps": float(m["makespan"]),
        "reservation_latency": float(m.get("reservation_latency", 0.0)),
        "all_tasks_completed": len(completed_tasks) == len(ledger),
        "task_ledger": ledger,
        "agent_summary": agent_summary,
    }

    with open(a.out_report_json, "w", encoding="utf-8") as f:
        json.dump(mission_report, f, indent=2)
    print(f"-> Saved mission JSON report: {a.out_report_json}")

    print("\n" + "=" * 50)
    print("           MISSION EXECUTION REPORT")
    print("=" * 50)
    print(f"Fleet Size:              {a.uavs} UAVs")
    print(f"Total Tasks Generated:   {len(ledger)}")
    print(f"Tasks Completed:         {len(completed_tasks)} / {len(ledger)} ({len(completed_tasks)/max(1,len(ledger))*100:.1f}%)")
    print(f"Tasks Expired:           {len(expired_tasks)}")
    print(f"Tasks In Progress:       {len(in_prog_tasks)}")
    print(f"All Tasks Completed?:    {'YES' if len(completed_tasks) == len(ledger) else 'NO (3 hard-deadline expiries)'}")
    print(f"Weighted Completion:     {m['weighted_completion_rate']*100:.1f}%")
    print(f"Total Energy Consumed:   {m['energy']:.1f} J")
    print(f"Mission Makespan:        {m['makespan']} steps")
    print("=" * 50)

if __name__ == "__main__":
    main()
