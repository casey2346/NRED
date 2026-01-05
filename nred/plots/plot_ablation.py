import matplotlib.pyplot as plt

labels = ["full", "no_latent", "no_consistency", "no_fallback"]
values = [0.18, 0.18, 0.11, 0.11]

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=["blue", "gray", "red", "red"])
plt.ylim(0, 1.0)
plt.title("Ablation Study Accuracy")
plt.ylabel("Accuracy")

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("bar_ablation.png")
print("Saved bar_ablation.png")
