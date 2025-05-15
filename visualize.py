from agent import df
import pandas as pd 
import matplotlib.pyplot as plt

# Plot cumulative rewards
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["cumulative_usa"], label="🇺🇸 USA", marker='o')
plt.plot(df["round"], df["cumulative_china"], label="🇨🇳 China", marker='s')
plt.title("Cumulative Rewards Over Rounds")
plt.xlabel("Round")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_trend.png")
plt.show()

action_counts = df[["usa_action", "china_action"]].apply(pd.Series.value_counts).fillna(0)
action_counts.plot(kind='bar', figsize=(8, 5), title="Action Frequency")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("action_frequency.png")
plt.show()
