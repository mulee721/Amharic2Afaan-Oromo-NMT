from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

<<<<<<< HEAD
=======

>>>>>>> 51da30f (Prepare NMT project for deployment)
# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"

<<<<<<< HEAD
HISTORY_CSV = PROJECT_ROOT / "checkpoints" / "training_history.csv"

FIGURES_DIR.mkdir(exist_ok=True)

=======

HISTORY_CSV = PROJECT_ROOT / "checkpoints" / "training_history.csv"


FIGURES_DIR.mkdir(exist_ok=True)


>>>>>>> 51da30f (Prepare NMT project for deployment)
# -------------------------
# Load training history
# -------------------------
df = pd.read_csv(HISTORY_CSV)
df = df.sort_values("epoch").reset_index(drop=True)

<<<<<<< HEAD
# -------------------------
=======

# -------------------------
# Set your test loss here
>>>>>>> 51da30f (Prepare NMT project for deployment)
# -------------------------
# Replace this with your actual test loss from evaluation
TEST_LOSS = 1.6959

<<<<<<< HEAD
# -------------------------
# Best validation epoch
# -------------------------

=======

# -------------------------
# Best validation epoch
# -------------------------
>>>>>>> 51da30f (Prepare NMT project for deployment)
valid_loss_series = pd.to_numeric(df["valid_loss"], errors="coerce")
best_idx = valid_loss_series.idxmin()
best_epoch = int(df.loc[best_idx, "epoch"])
best_valid_loss = float(valid_loss_series.loc[best_idx])

<<<<<<< HEAD
=======

>>>>>>> 51da30f (Prepare NMT project for deployment)
# -------------------------
# Plot: Train / Val / Test Loss
# -------------------------
plt.style.use("seaborn-v0_8-whitegrid")

<<<<<<< HEAD
fig, ax = plt.subplots(figsize=(8, 5))

=======

fig, ax = plt.subplots(figsize=(8, 5))


>>>>>>> 51da30f (Prepare NMT project for deployment)
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

<<<<<<< HEAD
=======

>>>>>>> 51da30f (Prepare NMT project for deployment)
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

<<<<<<< HEAD
=======

>>>>>>> 51da30f (Prepare NMT project for deployment)
# Test loss as horizontal line
ax.axhline(
    TEST_LOSS,
    color="#2ca02c",
    linewidth=2.6,
    linestyle="-.",
    label=f"Test loss = {TEST_LOSS:.4f}",
)

<<<<<<< HEAD
=======

>>>>>>> 51da30f (Prepare NMT project for deployment)
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

<<<<<<< HEAD
=======
# Annotate best validation point with numbers
ax.annotate(
    f"Epoch {best_epoch}\nLoss {best_valid_loss:.4f}",
    xy=(best_epoch, best_valid_loss),
    xytext=(5, 5),  # offset in points
    textcoords="offset points",
    fontsize=10,
    fontweight="600",
    color="#9467bd",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#9467bd", alpha=0.9),
)


>>>>>>> 51da30f (Prepare NMT project for deployment)
ax.set_xlabel("Epoch", fontsize=13, fontweight="600")
ax.set_ylabel("Loss", fontsize=13, fontweight="600")
ax.set_title(
    "Training, Validation, and Test Loss",
    fontsize=14,
    fontweight="700",
)

<<<<<<< HEAD
=======

>>>>>>> 51da30f (Prepare NMT project for deployment)
ax.legend(
    frameon=True,
    fontsize=11,
    loc="best",
)

<<<<<<< HEAD
ax.grid(True, which="both", linestyle="--", alpha=0.35)

fig.tight_layout()
output_path = FIGURES_DIR / "loss_train_val_test.png"
fig.savefig(output_path, dpi=300)

=======

ax.grid(True, which="both", linestyle="--", alpha=0.35)


fig.tight_layout()

output_path = FIGURES_DIR / "loss_train_val_test.png"

# Delete old image if it exists
if output_path.exists():
    output_path.unlink()

fig.savefig(output_path, dpi=300)


>>>>>>> 51da30f (Prepare NMT project for deployment)
plt.show()