# run_experiments.py
"""
Batch runner for wildfire firefighter simulation experiments.

Usage:
    mpirun -np 1 python run_experiments.py --runs 20
    mpirun -np 1 python run_experiments.py --runs 20 --param-file params.yaml --seed 100
"""

import argparse
import csv
import math
import os
from statistics import mean

from repast4py import parameters
from model import WildfireModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple wildfire simulation experiments.")
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of independent runs to execute (default: 10)",
    )
    parser.add_argument(
        "--param-file",
        type=str,
        default="params.yaml",
        help="YAML parameter file to use (default: params.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. If not provided, uses the seed in the param file.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="experiment_summary.csv",
        help="Output CSV file with per-run metrics (default: experiment_summary.csv)",
    )
    return parser.parse_args()


def run_single_experiment(param_file: str, base_seed: int, run_index: int):
    """
    Run a single simulation with a given seed and return a dict of metrics.
    """
    # Load params via repast4py helper
    p = parameters.init_params(param_file, "")

    # Get base seed from args or params.yaml
    if base_seed is None:
        base = int(p.get("random_seed", 42))
    else:
        base = base_seed

    # Derive a unique seed per run
    seed = base + run_index
    p["random_seed"] = seed

    # Put logs for this run in a unique subdirectory
    # (so the model's _clear_output_dir doesn't wipe other runs)
    p["log_dir"] = os.path.join("output", f"exp_run_{run_index:03d}")

    model = WildfireModel(p)
    model.run()

    # Collect metrics
    metrics = {
        "run_index": run_index,
        "seed": seed,
        "contained_at": model.contained_at if not math.isnan(model.contained_at) else None,
        "burning": model.count_burning(),
        "burnt": model.count_burnt(),
        "extinguished": model.count_extinguished(),
        "messages": model.messages,
        "firefighters": model.ff_count,
        "max_ticks": int(p["max_ticks"]),
    }

    return metrics


def main():
    args = parse_args()

    all_results = []

    print(f"Running {args.runs} experiments using '{args.param_file}' ...\n", flush=True)

    for i in range(args.runs):
        print(f"--- Run {i+1}/{args.runs} ---")
        metrics = run_single_experiment(args.param_file, args.seed, i)
        all_results.append(metrics)

        ca = metrics["contained_at"]
        ca_str = "nan" if ca is None else f"{ca:.2f}"
        print(
            f"Run {i}: seed={metrics['seed']} "
            f"contained_at={ca_str} "
            f"burnt={metrics['burnt']} "
            f"extinguished={metrics['extinguished']} "
            f"messages={metrics['messages']}",
        flush=True)
        print()

    # Write per-run summary CSV
    fieldnames = [
        "run_index",
        "seed",
        "contained_at",
        "burning",
        "burnt",
        "extinguished",
        "messages",
        "firefighters",
        "max_ticks",
    ]

    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    # Compute aggregate stats
    contained_times = [r["contained_at"] for r in all_results if r["contained_at"] is not None]
    burnt_vals = [r["burnt"] for r in all_results]
    ext_vals = [r["extinguished"] for r in all_results]
    msg_vals = [r["messages"] for r in all_results]

    print("=== Aggregate Results ===")

    print(f"Total runs: {len(all_results)}")
    print(f"Contained runs: {len(contained_times)}")
    if contained_times:
        print(f"Average contained_at: {mean(contained_times):.2f}")
        print(f"Min contained_at: {min(contained_times):.2f}")
        print(f"Max contained_at: {max(contained_times):.2f}")
    else:
        print("No runs reached full containment.")

    print(f"Average final burnt cells: {mean(burnt_vals):.2f}")
    print(f"Average final extinguished cells: {mean(ext_vals):.2f}")
    print(f"Average total messages: {mean(msg_vals):.2f}")

    print(f"\nPer-run metrics written to: {args.summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
