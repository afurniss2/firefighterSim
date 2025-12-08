
import os
from repast4py import parameters

from model import WildfireModel
from experiment4_logger import Experiment4Logger


FIREFIGHTER_VALUES = [4, 8, 12, 16, 24, 32, 40]

N_RUNS_PER_SETTING = 50

BASE_PARAM_FILE = "params.yaml"
OUTPUT_DIR = "output_exp4"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "exp4_summary.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = Experiment4Logger(SUMMARY_CSV)

    for ff_count in FIREFIGHTER_VALUES:
        print(f"\n=== Experiment 4: firefighters={ff_count} ===")
        for run_idx in range(N_RUNS_PER_SETTING):
            print(f"  Run {run_idx + 1}/{N_RUNS_PER_SETTING}")

            p = parameters.init_params(BASE_PARAM_FILE, "")

            p["width"] = 50
            p["height"] = 50
            p["ignitions"] = 3
            p["base_spread_p"] = 0.4
            p["firefighters"] = ff_count
            p["perception_r"] = 6
            p["comm_r"] = 6
            p["comm_freq"] = 2

            base_seed = int(p.get("random_seed", 42))
            seed = base_seed + run_idx
            p["random_seed"] = seed

            p["log_dir"] = os.path.join(
                OUTPUT_DIR,
                f"ff_{ff_count}_run_{run_idx:03d}"
            )

            model = WildfireModel(p)
            model.run()

            logger.log_run(run_idx, model, seed)

    print(f"\nExperiment 4 complete. Summary written to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
