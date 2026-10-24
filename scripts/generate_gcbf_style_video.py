#!/usr/bin/env python
"""Generate high-resolution 3D isometric video matching GCBF SimpleDrone visualization standards.
- 2000x2000 (10x10 inches @ 200 DPI)
- Clean white background, crisp isometric projection
- Agent spheres with distinct colors, black outlines, inner white agent ID numbers
- Smooth dashed trajectory trails from initial takeoff to target
- Star markers for task/goal locations (green for completed, colored for active) with labels
- Dynamic real-time HUD telemetry at top-left:
    Timestep: t / T
    Agents: N | Active Comms Links: L | Min Dist: D
    Tasks Completed: C / Total (Rate: %) | Fleet Energy: E J
- High-quality H.264 MP4 export via ffmpeg at 20 fps

Usage:
    python scripts/generate_gcbf_style_video.py --uavs 20 --tasks 50 --fps 20 --dpi 200
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import _common  # noqa: F401

from uav_allocation.config import load_config
from uav_allocation.environment.simulator import UAVTaskEnv
from uav_allocation.algorithms.rollout import run_episode
from uav_allocation.evaluation.evaluate import make_policy
from uav_allocation.visualization.simulator_view_3d import AltitudeTracker

def main():
    parser = argparse.ArgumentParser(description="Generate GCBF-style clean 3D isometric video")
    parser.add_argument("--method", default="gnn_mappo")
    parser.add_argument("--uavs", type=int, default=20)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2_000_100)
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--fps", type=int, default=20, help="Video framerate (default 20)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for video generation (default 150 = 1500x1500px, 200 = 2000x2000px)")
    parser.add_argument("--save-video", type=str, default="results/uav_simulation_3d_isometric.mp4")
    parser.add_argument("--save-frame", type=str, default="results/uav_simulation_3d_isometric.png")
    args = parser.parse_args()

    print(f"Initializing simulation: {args.method}, {args.uavs} UAVs, {args.tasks} tasks...")
    cfg = load_config(overrides=[f"env.n_uavs={args.uavs}", f"env.n_tasks={args.tasks}"])
    pol, cfg = make_policy(args.method, cfg, args.checkpoint)
    env = UAVTaskEnv(cfg)

    tracker = AltitudeTracker(args.uavs, arena_size=cfg.env.arena)
    history = []
    task_catalog = {}

    def on_step(e):
        tracker.step(e)

        # Collect current 3D positions for all UAVs
        uav_pos = np.array([
            [float(u.pos[0]), float(u.pos[1]), float(tracker.current_altitudes[u.id])]
            for u in e.uavs
        ])
        socs = np.array([float(u.soc) for u in e.uavs])
        statuses = np.array([int(u.status) for u in e.uavs])
        task_ids = np.array([int(u.task_id) for u in e.uavs])

        # Track pairwise minimum distance among UAVs
        pos_diff = uav_pos[:, None, :] - uav_pos[None, :, :]
        dists = np.linalg.norm(pos_diff, axis=2)
        np.fill_diagonal(dists, 1e6)
        min_dist = float(np.min(dists))

        # Register tasks
        for t in e.tasks:
            if t.announced:
                task_catalog[t.id] = {
                    "pos": np.array([float(t.pos[0]), float(t.pos[1]), float(tracker.get_task_altitude(t))]),
                    "priority": int(t.priority),
                    "status": t.status.name,
                    "owner": int(t.owner),
                }

        # Count active communication links
        active_links = int(np.sum(e.coord.adj) // 2) if e.mode != "centralized" else 0
        done_cnt = sum(t.status.name == "COMPLETED" for t in e.tasks)
        total_announced = sum(t.announced for t in e.tasks)

        history.append({
            "t": int(e.t),
            "uav_pos": uav_pos,
            "socs": socs,
            "statuses": statuses,
            "task_ids": task_ids,
            "min_dist": min_dist,
            "active_links": active_links,
            "done_cnt": done_cnt,
            "total_announced": total_announced,
            "energy": float(e.m.get("energy", 0.0)),
        })

    print("Running simulation episode rollout...")
    m = run_episode(env, pol, args.seed, on_step=on_step, frame_every=1)
    print(f"Episode completed: {len(history)} steps. Weighted completion: {m['weighted_completion_rate']*100:.1f}%")

    print(f"Setting up clean 3D isometric figure (size 10x10 inches @ {args.dpi} DPI = {10*args.dpi}x{10*args.dpi}px)...")
    fig = plt.figure(figsize=(10, 10), dpi=args.dpi, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Color palette matching GCBF (distinct colors per agent)
    cmap = matplotlib.colormaps['tab20'].resampled(args.uavs)
    agent_colors = [cmap(i) for i in range(args.uavs)]

    # Draw Bases on ground plane
    for i, b in enumerate(env.bases):
        ax.scatter([b[0]], [b[1]], [0], marker='^', s=160, c='black', edgecolors='green', lw=1.5, zorder=5)
        ax.text(b[0] + 18, b[1] + 18, 2, f"Base {i}", fontsize=9, color='black', weight='bold')

    # Plot Goals / Tasks (star markers like GCBF)
    task_scatters = {}
    task_labels = {}
    for tid, tinfo in task_catalog.items():
        p = tinfo["pos"]
        # Star marker
        st = ax.scatter([p[0]], [p[1]], [p[2]], marker='*', s=140, c='green', alpha=0.85, zorder=4)
        lbl = ax.text(p[0], p[1], p[2] + 4.0, f"T{tid}", fontsize=8, color='#15803d', ha='center', weight='bold')
        task_scatters[tid] = st
        task_labels[tid] = lbl

    # Plot Obstacles / No-Fly Zones if any
    for z in env.cfg.env.no_fly_zones:
        xc, yc, r = float(z[0]), float(z[1]), float(z[2])
        theta = np.linspace(0, 2*np.pi, 20)
        z_cyl = np.linspace(0, 180, 6)
        Theta, Z_grid = np.meshgrid(theta, z_cyl)
        X_grid = xc + r * np.cos(Theta)
        Y_grid = yc + r * np.sin(Theta)
        ax.plot_surface(X_grid, Y_grid, Z_grid, color='gray', alpha=0.25, shade=False)

    # Scatters, trails, and text labels for agents
    scatters = []
    trails = []
    texts = []
    drop_lines = []

    for i in range(args.uavs):
        color = agent_colors[i]
        # Circular drone sphere
        scat = ax.scatter([], [], [], c=[color], s=260, marker='o', edgecolors='black', linewidths=1.2, zorder=8)
        scatters.append(scat)
        # Dashed history trail
        trail, = ax.plot([], [], [], c=color, alpha=0.6, linestyle='--', linewidth=1.5)
        trails.append(trail)
        # Drop line to ground
        dline, = ax.plot([], [], [], c='gray', alpha=0.3, linestyle=':', linewidth=0.8)
        drop_lines.append(dline)
        # Agent ID text centered on sphere
        txt = ax.text(0, 0, 0, f"{i}", fontsize=8.5, ha='center', va='center', zorder=12, color='white', weight='bold')
        texts.append(txt)

    # Clean axis limits matching GCBF arena grid
    A = float(cfg.env.arena)
    ax.set_xlim(0, A)
    ax.set_ylim(0, A)
    ax.set_zlim(0, 180)
    ax.view_init(elev=28, azim=45)

    # Clean grid styling
    ax.set_xlabel('X (m)', fontsize=10, labelpad=6)
    ax.set_ylabel('Y (m)', fontsize=10, labelpad=6)
    ax.set_zlabel('Altitude Z (m)', fontsize=10, labelpad=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, linestyle='-', alpha=0.4)

    # Title & Legend matching GCBF
    ax.set_title("Decentralized UAV Dynamic Task Allocation 3D Simulation", fontsize=15, weight='bold', pad=15)
    
    # Custom legend items
    proxy_agent = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0284c7', markeredgecolor='black', markersize=10, label='UAV Agents')
    proxy_goal = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markeredgecolor='green', markersize=14, label='Tasks / Goals')
    proxy_base = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markeredgecolor='green', markersize=10, label='Charging Bases')
    ax.legend(handles=[proxy_agent, proxy_goal, proxy_base], loc='upper right', fontsize=10, framealpha=0.9)

    # Telemetry HUD in top-left matching GCBF format
    time_text = fig.text(0.06, 0.95, '', fontsize=11, weight='bold')
    status_text = fig.text(0.06, 0.91, '', fontsize=10)

    # Pre-extract all agent positions for fast indexing
    all_uav_pos = np.array([h["uav_pos"] for h in history]) # (T, N, 3)

    def init():
        for scat in scatters:
            scat._offsets3d = ([], [], [])
        for trail in trails:
            trail.set_data([], [])
            trail.set_3d_properties([])
        for dline in drop_lines:
            dline.set_data([], [])
            dline.set_3d_properties([])
        for text in texts:
            text.set_position((0, 0))
            text.set_3d_properties(0, 'z')
            text.set_text("")
        time_text.set_text('')
        status_text.set_text('')
        return scatters + trails + drop_lines + texts + [time_text, status_text]

    def update(frame):
        step_data = history[frame]
        pos = step_data["uav_pos"]

        for i in range(args.uavs):
            # Update agent sphere position
            scatters[i]._offsets3d = ([pos[i, 0]], [pos[i, 1]], [pos[i, 2]])

            # Update historical trail (last 60 steps for clean visualization)
            s_start = max(0, frame - 60)
            hist = all_uav_pos[s_start : frame + 1, i, :]
            trails[i].set_data(hist[:, 0], hist[:, 1])
            trails[i].set_3d_properties(hist[:, 2])

            # Drop line to ground
            drop_lines[i].set_data([pos[i, 0], pos[i, 0]], [pos[i, 1], pos[i, 1]])
            drop_lines[i].set_3d_properties([0, pos[i, 2]])

            # Update Agent ID label centered inside sphere
            texts[i].set_position((pos[i, 0], pos[i, 1]))
            texts[i].set_3d_properties(pos[i, 2], 'z')
            texts[i].set_text(f"{i}")

        # Update HUD text
        t = step_data["t"]
        total_steps = len(history) - 1
        time_text.set_text(f"Timestep: {t} / {total_steps}")

        done = step_data["done_cnt"]
        tot = step_data["total_announced"]
        rate = (done / max(1, tot)) * 100
        min_d = step_data["min_dist"]
        links = step_data["active_links"]
        energy = step_data["energy"]

        status_text.set_text(
            f"Agents: {args.uavs} | Min Inter-UAV Dist: {min_d:.1f}m | Mesh RF Links: {links}\n"
            f"Tasks Completed: {done} / {tot} ({rate:.1f}%) | Energy: {energy:.1f} J"
        )
        status_text.set_color('#166534' if done >= 40 else '#1e293b')

        return scatters + trails + drop_lines + texts + [time_text, status_text]

    print(f"Compiling animation ({len(history)} frames @ {args.fps} fps)...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(history), init_func=init, blit=False, interval=int(1000 / args.fps)
    )

    out_path = Path(args.save_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving high-resolution video to {out_path} using ffmpeg...")
    ani.save(str(out_path), writer='ffmpeg', fps=args.fps, bitrate=2500)
    print(f"-> Video successfully saved: {out_path}")

    # Also save the final frame high-res image
    fig.savefig(args.save_frame, dpi=args.dpi, bbox_inches='tight')
    print(f"-> Saved reference snapshot frame: {args.save_frame}")
    plt.close(fig)

if __name__ == "__main__":
    main()
