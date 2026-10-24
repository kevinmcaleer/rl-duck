"""Draw the reward curve from logs/monitor.csv (written by ladder.py) to reward_curve.png."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/monitor.csv", skiprows=1)   # columns: r (score), l (length), t (seconds)
df["steps"] = df["l"].cumsum()
plt.figure(figsize=(16, 9), dpi=120)
plt.plot(df["steps"], df["r"], alpha=0.3, label="each attempt")
plt.plot(df["steps"], df["r"].rolling(20, min_periods=1).mean(), linewidth=4, label="average")
plt.xlabel("Training steps", fontsize=20)
plt.ylabel("Score per attempt", fontsize=20)
plt.ylim(0, 1050)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("reward_curve.png")
print("Saved reward_curve.png")