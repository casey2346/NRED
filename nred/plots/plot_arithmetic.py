import matplotlib.pyplot as plt

baseline = 0.18
nred = 0.18

plt.figure(figsize=(4, 4))
plt.bar(["Baseline", "NRED"], [baseline, nred], color=["gray", "blue"])
plt.ylim(0, 1.0)
plt.title("Arithmetic Task Accuracy")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("bar_arithmetic.png")
print("Saved bar_arithmetic.png")
