
import os
from repast4py import parameters

from model import WildfireModel
from experiment3_logger import Experiment3Logger


# Perception radius values
PERCEPTION_VALUES = [2, 3, 4, 6, 8, 10]

# How many runs per setting
N_RUNS_PER_SETTING = 50  

BASE_PARAM_FILE = "params.yaml"
OUTPUT_DIR = "output_exp3"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "exp3_summary.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = Experiment3Logger(SUMMARY_CSV)

    for r in PERCEPTION_VALUES:
        print(f"\n=== Experiment 3: perception_r={r} ===")
        for run_idx in range(N_RUNS_PER_SETTING):
            print(f"  Run {run_idx + 1}/{N_RUNS_PER_SETTING}")

            p = parameters.init_params(BASE_PARAM_FILE, "")

            p["width"] = 50
            p["height"] = 50
            p["ignitions"] = 3
            p["base_spread_p"] = 0.4
            p["firefighters"] = 24
            p["comm_freq"] = 2
            p["comm_r"] = 6
            p["perception_r"] = r

            # Seed and log_dir per run
            base_seed = int(p.get("random_seed", 42))
            seed = base_seed + run_idx
            p["random_seed"] = seed

            p["log_dir"] = os.path.join(
                OUTPUT_DIR,
                f"perception_{r}_run_{run_idx:03d}"
            )

            model = WildfireModel(p)
            model.run()

            logger.log_run(run_idx, model, seed)

    print(f"\nExperiment 3 complete. Summary written to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
