#!/usr/bin/env python
"""Visualise an episode (GIF + final PNG in 2D or 3D).

Examples:
    python scripts/run_simulation.py --method gnn_mappo --uavs 20 --tasks 50 --checkpoint checkpoints/best.pt
    python scripts/run_simulation.py --method gnn_mappo --3d --checkpoint checkpoints/best.pt
"""
import argparse
import io
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import _common  # noqa: F401
from uav_allocation.config import load_config
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.algorithms.rollout import run_episode
from uav_allocation.evaluation.evaluate import make_policy
from uav_allocation.visualization.simulator_view import render_frame

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="gnn_mappo")
ap.add_argument("--uavs", type=int, default=20)
ap.add_argument("--tasks", type=int, default=50)
ap.add_argument("--seed", type=int, default=2_000_100)
ap.add_argument("--checkpoint", default="checkpoints/best.pt")
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--out", default="results/simulation.gif")
ap.add_argument("--3d", "--view3d", dest="view_3d", action="store_true", help="Generate 3D visualization outputs")
ap.add_argument("-o", "--override", action="append", default=[])
a = ap.parse_args()

cfg = load_config(overrides=[f"env.n_uavs={a.uavs}", f"env.n_tasks={a.tasks}"] + a.override)
pol, cfg = make_policy(a.method, cfg, a.checkpoint)

if a.view_3d:
    from uav_allocation.visualization.simulator_view_3d import (
        AltitudeTracker,
        render_frame_3d,
        render_trajectories_3d,
        render_flight_profiles,
        generate_interactive_3d_html,
    )
    tracker = AltitudeTracker(a.uavs, arena_size=cfg.env.arena)
    fig = plt.figure(figsize=(8, 7), dpi=90)
    ax = fig.add_subplot(111, projection="3d")
    frames = []

    def hook(env):
        tracker.step(env)
        azim = 35.0 + (env.t * 0.45) % 360.0
        b = io.BytesIO()
        render_frame_3d(env, ax, tracker, title=a.method, azim=azim)
        fig.savefig(b, format="png", dpi=85, bbox_inches="tight")
        b.seek(0)
        frames.append(Image.open(b).convert("P"))

    env = UAVTaskEnv(cfg)
    m = run_episode(env, pol, a.seed, on_step=hook, frame_every=a.every)
    plt.close(fig)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=150, loop=0)
    frames[-1].convert("RGB").save(str(Path(a.out).with_suffix(".png")))

    # Also save trajectories and HTML
    html_out = str(Path(a.out).with_suffix(".html"))
    traj_out = str(Path(a.out).parent / "trajectories_3d.png")
    prof_out = str(Path(a.out).parent / "flight_profiles_3d.png")
    render_trajectories_3d(tracker, env, traj_out)
    render_flight_profiles(tracker, env, prof_out)
    generate_interactive_3d_html(tracker, env, html_out, method_name=a.method.upper())
    print(f"saved 3D visual outputs: {a.out}, {traj_out}, {prof_out}, {html_out}")
else:
    fig, ax = plt.subplots(figsize=(6, 6))
    frames = []
    def hook(env):
        b = io.BytesIO()
        render_frame(env, ax, title=a.method)
        fig.savefig(b, format="png", dpi=70)
        b.seek(0)
        frames.append(Image.open(b).convert("P"))

    m = run_episode(UAVTaskEnv(cfg), pol, a.seed, on_step=hook, frame_every=a.every)
    plt.close(fig)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=150, loop=0)
    frames[-1].convert("RGB").save(str(Path(a.out).with_suffix(".png")))
    print(f"saved {a.out} ({len(frames)} frames); weighted_completion={m['weighted_completion_rate']:.3f}")
