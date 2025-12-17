import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output_exp3/exp3_summary.csv")

# Group by perception radius
g_mean = df.groupby("perception_r").mean()
g_median = df.groupby("perception_r").median()

print("=== Experiment 3 Summary (means by perception_r) ===")
print(g_mean)

# Plot containment time
plt.figure()
plt.plot(g_mean.index, g_mean["contained_at"], marker="o", label="Avg")
plt.plot(g_median.index, g_median["contained_at"], marker="o", linestyle="--", label="Median")
plt.title("Exp 3: Containment Time vs Perception Radius")
plt.xlabel("Perception Radius")
plt.ylabel("Ticks to Containment")
plt.legend()
plt.grid(True)
plt.savefig("exp3_containment_time.png")

plt.figure()
plt.plot(g_mean.index, g_mean["final_burnt"], marker="o", label="Avg")
plt.plot(g_median.index, g_median["final_burnt"], marker="o", linestyle="--", label="Median")
plt.title("Exp 3: Burned Area vs Perception Radius")
plt.xlabel("Perception Radius")
plt.ylabel("Final Burned Cells")
plt.legend()
plt.grid(True)
plt.savefig("exp3_burned_area.png")
