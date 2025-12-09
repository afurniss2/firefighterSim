
import csv
import os
import math


class Experiment3Logger:
    """
    Logger for Experiment 3: effect of perception radius.

    """

    def __init__(self, summary_csv: str):
        self.summary_csv = summary_csv

        directory = os.path.dirname(summary_csv)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.summary_csv):
            with open(self.summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_index",
                    "seed",
                    "perception_r",
                    "final_burnt",
                    "contained_at",
                    "idle_pct",
                    "avg_cells_scanned",
                    "messages",
                    "firefighters",
                    "max_ticks",
                ])

    def log_run(self, run_index: int, model, seed: int):
        """Call this AFTER model.run()."""

        decisions = model.perception_decisions
        idle_decisions = model.perception_idle_decisions
        cells_scanned = model.perception_cells_scanned

        if decisions > 0:
            idle_pct = idle_decisions / decisions
            avg_cells_scanned = cells_scanned / decisions
        else:
            idle_pct = 0.0
            avg_cells_scanned = 0.0

        contained_at = model.contained_at
        if isinstance(contained_at, float) and math.isnan(contained_at):
            contained_at_val = ""
        else:
            contained_at_val = contained_at

        with open(self.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_index,
                seed,
                model.perception_r,
                model.count_burnt(),
                contained_at_val,
                idle_pct,
                avg_cells_scanned,
                model.messages,
                model.ff_count,
                model.max_ticks,
            ])
