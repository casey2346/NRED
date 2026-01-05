import matplotlib.pyplot as plt

baseline = 1.0
nred = 1.0

plt.figure(figsize=(4, 4))
plt.bar(["Baseline", "NRED"], [baseline, nred], color=["gray", "blue"])
plt.ylim(0, 1.1)
plt.title("Parity Task Accuracy")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("bar_parity.png")
print("Saved bar_parity.png")
