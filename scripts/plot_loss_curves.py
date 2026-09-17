from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"

HISTORY_CSV = PROJECT_ROOT / "checkpoints" / "training_history.csv"

FIGURES_DIR.mkdir(exist_ok=True)

# -------------------------
# Load training history
# -------------------------
df = pd.read_csv(HISTORY_CSV)
df = df.sort_values("epoch").reset_index(drop=True)

# -------------------------
# -------------------------
# Replace this with your actual test loss from evaluation
TEST_LOSS = 1.6959

# -------------------------
# Best validation epoch
# -------------------------

valid_loss_series = pd.to_numeric(df["valid_loss"], errors="coerce")
best_idx = valid_loss_series.idxmin()
best_epoch = int(df.loc[best_idx, "epoch"])
best_valid_loss = float(valid_loss_series.loc[best_idx])

# -------------------------
# Plot: Train / Val / Test Loss
# -------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(8, 5))

# Training loss
ax.plot(
    df["epoch"],
    pd.to_numeric(df["train_loss"], errors="coerce"),
    label="Training loss",
    color="#1f77b4",
    linewidth=2.4,
    marker="o",
    markersize=4,
)

# Validation loss
ax.plot(
    df["epoch"],
    valid_loss_series,
    label="Validation loss",
    color="#d62728",
    linewidth=2.4,
    marker="s",
    markersize=5,
    linestyle="--",
)

# Test loss as horizontal line
ax.axhline(
    TEST_LOSS,
    color="#2ca02c",
    linewidth=2.6,
    linestyle="-.",
    label=f"Test loss = {TEST_LOSS:.4f}",
)

# Best validation epoch marker
ax.scatter(
    [best_epoch],
    [best_valid_loss],
    color="#9467bd",
    s=90,
    zorder=4,
    label=f"Best validation (epoch {best_epoch})",
    edgecolor="black",
    linewidth=1.2,
)

ax.set_xlabel("Epoch", fontsize=13, fontweight="600")
ax.set_ylabel("Loss", fontsize=13, fontweight="600")
ax.set_title(
    "Training, Validation, and Test Loss",
    fontsize=14,
    fontweight="700",
)

ax.legend(
    frameon=True,
    fontsize=11,
    loc="best",
)

ax.grid(True, which="both", linestyle="--", alpha=0.35)

fig.tight_layout()
output_path = FIGURES_DIR / "loss_train_val_test.png"
fig.savefig(output_path, dpi=300)

plt.show()