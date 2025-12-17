import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output_exp4/exp4_summary.csv")

g_mean = df.groupby("firefighters").mean()
g_median = df.groupby("firefighters").median()

print("=== Experiment 4 Summary (means by firefighter count) ===")
print(g_mean)

# Containment time
plt.figure()
plt.plot(g_mean.index, g_mean["contained_at"], marker="o", label="Avg")
plt.plot(g_median.index, g_median["contained_at"], marker="o", linestyle="--", label="Median")
plt.title("Exp 4: Containment Time vs Firefighter Count")
plt.xlabel("Number of Firefighters")
plt.ylabel("Ticks to Containment")
plt.legend()
plt.grid(True)
plt.savefig("exp4_containment_time.png")

# Burned area
plt.figure()
plt.plot(g_mean.index, g_mean["final_burnt"], marker="o", label="Avg")
plt.plot(g_median.index, g_median["final_burnt"], marker="o", linestyle="--", label="Median")
plt.title("Exp 4: Burned Area vs Firefighter Count")
plt.xlabel("Number of Firefighters")
plt.ylabel("Final Burned Cells")
plt.legend()
plt.grid(True)
plt.savefig("exp4_burned_area.png")
