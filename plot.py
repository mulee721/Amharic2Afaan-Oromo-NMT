import json
from pathlib import Path
import matplotlib.pyplot as plt

# File paths
JSON_PATH = Path(
    r"C:\Users\Mulee\Documents\AI-MT\checkpoints\training_history.json"
)
FIGURE_DIR = Path(r"C:\Users\Mulee\Documents\AI-MT\figures")
OUTPUT_IMAGE = FIGURE_DIR / "loss_train_val.png"


def plot_loss_history():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON history file not found at: {JSON_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Parse JSON history keys correctly ("valid_loss")
    epochs = []
    train_loss = []
    valid_loss = []

    for idx, item in enumerate(history):
        epochs.append(item.get("epoch", idx + 1))
        train_loss.append(item.get("train_loss") or item.get("loss"))
        valid_loss.append(
            item.get("valid_loss")
            if item.get("valid_loss") is not None
            else item.get("val_loss")
        )

    # Identify the Best Epoch (minimum validation loss)
    min_val_idx = valid_loss.index(min(valid_loss))
    best_epoch = epochs[min_val_idx]
    best_val_loss = valid_loss[min_val_idx]
    best_train_loss = train_loss[min_val_idx]

    # Create figure
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot Train and Validation Loss
    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
        color="#1f77b4",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        epochs,
        valid_loss,
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2,
        marker="s",
        markersize=4,
    )

    # Draw vertical dashed line for the best epoch
    plt.axvline(
        x=best_epoch,
        color="#2ca02c",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch ({best_epoch})",
    )

    # Mark lowest validation loss point with a red star
    plt.plot(
        best_epoch,
        best_val_loss,
        marker="*",
        markersize=12,
        color="#d62728",
        label=f"Min Val Loss ({best_val_loss:.4f})",
    )

    # Add text annotation callout box
    annotation_text = (
        f"★ Best Epoch: {best_epoch}\n"
        f"Val Loss: {best_val_loss:.4f}\n"
        f"Train Loss: {best_train_loss:.4f}"
    )

    plt.annotate(
        annotation_text,
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch - 7, best_val_loss + 0.65),
        arrowprops=dict(
            facecolor="black", shrink=0.08, width=1, headwidth=6
        ),
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#ffffd8",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
        fontsize=10,
        fontweight="bold",
    )

    plt.title(
        "Training & Validation Loss vs. Epoch",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    # Save output
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    plt.close()

    print(f"✓ Loss graph successfully saved to: {OUTPUT_IMAGE}")
    print(
        f"  ★ Best Epoch: {best_epoch} | Valid Loss: {best_val_loss:.6f} | Train Loss: {best_train_loss:.6f}"
    )


if __name__ == "__main__":
    plot_loss_history()