
import csv
import os
import math


class Experiment4Logger:
    """
    Logger for Experiment 4: effect of firefighter team size.

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
                    "firefighters",
                    "final_burnt",
                    "contained_at",
                    "run_length",
                    "collision_events",
                    "collisions_per_tick",
                    "messages",
                    "perception_r",
                    "comm_r",
                    "max_ticks",
                ])

    def log_run(self, run_index: int, model, seed: int):
        """Call this AFTER model.run()."""

        contained_at = model.contained_at
        if isinstance(contained_at, float) and math.isnan(contained_at):
            contained_val = ""
            run_length = model.max_ticks
        else:
            contained_val = contained_at
            run_length = contained_at

        if run_length and run_length > 0:
            collisions_per_tick = model.collision_events / run_length
        else:
            collisions_per_tick = 0.0

        with open(self.summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_index,
                seed,
                model.ff_count,
                model.count_burnt(),
                contained_val,
                run_length,
                model.collision_events,
                collisions_per_tick,
                model.messages,
                model.perception_r,
                model.comm_r,
                model.max_ticks,
            ])
