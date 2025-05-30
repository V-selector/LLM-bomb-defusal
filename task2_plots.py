import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/sandra/desktop/xyy.lee15/llm-bomb-defusal/task2_results.csv")
df["top_k"] = df["top_k"].fillna("None").astype(str)

# Chart 1
prompt_order = ['text', 'markdown', 'json']
prompt_steps = df.groupby("prompt_style")["steps"].mean().reset_index()
prompt_steps["prompt_style"] = pd.Categorical(prompt_steps["prompt_style"], categories=prompt_order, ordered=True)
prompt_steps = prompt_steps.sort_values("prompt_style")
plt.figure(figsize=(8, 5))
plt.bar(prompt_steps["prompt_style"], prompt_steps["steps"], color='skyblue')
plt.xlabel("Prompt Style", fontsize=12, fontweight='bold',labelpad=10)
plt.ylabel("Average Steps", fontsize=12, fontweight='bold',labelpad=10)
plt.title("Prompt Style vs Average Steps", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(axis='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("/Users/sandra/desktop/xyy.lee15/task2_plots/prompt_style_vs_avg_steps.png")
plt.show()

# Chart 2
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="top_k", y="steps", hue="prompt_style", marker="o", linewidth=2)
plt.xlabel("Top-k", fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel("Average Steps", fontsize=12, fontweight='bold', labelpad=10)
plt.title("Top-k vs Average Steps by Prompt Style", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("/Users/sandra/desktop/xyy.lee15/task2_plots/topk_vs_steps_by_prompt.png")
plt.show()

# Chart 3
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="temperature", y="success", hue="prompt_style", marker="o", linewidth=2)
plt.xlabel("Temperature", fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel("Success Rate", fontsize=12, fontweight='bold', labelpad=10)
plt.title("Temperature vs Success Rate by Prompt Style", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tick_params(axis='both', direction='in', length=6)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("/Users/sandra/desktop/xyy.lee15/task2_plots/temperature_vs_success_by_prompt.png")
plt.show()
