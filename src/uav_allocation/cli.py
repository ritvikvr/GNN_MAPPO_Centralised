"""CLI Entry point for UAV Task Allocation Simulation and Evaluation."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="uav-sim",
        description="Decentralized UAV Dynamic Task Allocation using GNN + MAPPO",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: simulate
    sim_parser = subparsers.add_parser("simulate", help="Run 3D isometric simulation and render video")
    sim_parser.add_argument("--steps", type=int, default=300, help="Simulation steps (default: 300)")
    sim_parser.add_argument("--fps", type=int, default=10, help="Output video frames per second (default: 10)")
    sim_parser.add_argument("--output", type=str, default="results/uav_simulation_3d_isometric.mp4", help="Output video file path")
    sim_parser.add_argument("--snapshot", type=str, default="results/uav_simulation_3d_isometric.png", help="Output snapshot image path")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Generate comparative plots against baselines")
    bench_parser.add_argument("--output-dir", type=str, default="results", help="Directory to save comparison plots")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained policy on test scenarios")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    eval_parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "simulate":
        from scripts.generate_gcbf_style_video import render_isometric_simulation
        print(f"Starting 3D isometric simulation ({args.steps} steps @ {args.fps} fps)...")
        render_isometric_simulation(
            output_video_path=args.output,
            output_snapshot_path=args.snapshot,
            total_steps=args.steps,
            fps=args.fps,
        )
        print("Simulation video rendering completed successfully.")

    elif args.command == "benchmark":
        from scripts.generate_comparison_plots import main as generate_plots
        print(f"Generating benchmark comparison plots in '{args.output_dir}'...")
        generate_plots()
        print("Benchmark plots generated successfully.")

    elif args.command == "evaluate":
        from uav_allocation.evaluation.evaluator import Evaluator
        print(f"Evaluating checkpoint '{args.checkpoint}' over {args.episodes} episodes...")
        evaluator = Evaluator(checkpoint_path=args.checkpoint)
        metrics = evaluator.run(num_episodes=args.episodes)
        print(f"Evaluation complete. Metrics summary: {metrics}")


if __name__ == "__main__":
    main()
