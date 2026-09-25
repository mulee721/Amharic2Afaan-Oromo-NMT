from __future__ import annotations
import argparse
import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import Batch, create_dataloaders
from transformer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    TransformerNMT,
)


DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_MAX_LENGTH = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_GRAD_CLIP = 1.0


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Select CPU, CUDA, or automatic device."""
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but no CUDA GPU is available."
        )

    return device


def format_time(seconds: float) -> str:
    """Format seconds as a readable duration."""
    if not math.isfinite(seconds):
        return "--"

    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"

    return f"{seconds}s"


def format_number(value: int | float) -> str:
    """Format a number with thousands separators."""
    return f"{value:,}"


def calculate_token_count(
    target_labels: torch.Tensor,
) -> int:
    """Count target labels that are not padding."""
    return int(
        target_labels.ne(PAD_ID).sum().item()
    )


def calculate_perplexity(loss: float) -> float:
    """Safely calculate perplexity from cross-entropy loss."""
    return math.exp(min(loss, 20.0))


def create_grad_scaler(
    device: torch.device,
) -> torch.amp.GradScaler | None:
    """
    Create a CUDA gradient scaler only when CUDA is available.

    CPU training does not need gradient scaling.
    """
    if device.type != "cuda":
        return None

    return torch.amp.GradScaler(
        "cuda",
        enabled=True,
    )
def run_epoch(
    model: TransformerNMT,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP,
    epoch: int = 1,
    total_epochs: int = 1,
    phase: str = "TRAINING",
    show_progress: bool = True,
) -> tuple[float, float, float, int]:
    """
    Run one complete training or validation epoch.

    Prints one clean progress line every 250 batches.
    This avoids terminal corruption in PowerShell.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    epoch_start = time.perf_counter()
    total_batches = len(loader)

    print()
    print(
        f"{phase} STARTED | "
        f"EPOCH {epoch:02d}/{total_epochs:02d} | "
        f"TOTAL BATCHES {total_batches:,}"
    )

    context = (
        torch.enable_grad()
        if is_training
        else torch.inference_mode()
    )

    with context:
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            batch: Batch = batch.to(device)

            target_input = batch.target[:, :-1]
            target_labels = batch.target[:, 1:]
            target_padding_mask = target_input.eq(
                PAD_ID
            )

            if is_training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            use_amp = (
                scaler is not None
                and device.type == "cuda"
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(
                    source=batch.source,
                    target_input=target_input,
                    source_padding_mask=(
                        batch.source_padding_mask
                    ),
                    target_padding_mask=(
                        target_padding_mask
                    ),
                )

                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    target_labels.reshape(-1),
                )

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip_norm,
                    )

                    optimizer.step()

            token_count = calculate_token_count(
                target_labels
            )

            total_loss += (
                loss.item() * token_count
            )

            total_tokens += token_count

            running_loss = (
                total_loss
                / max(total_tokens, 1)
            )

            running_perplexity = (
                calculate_perplexity(running_loss)
            )

            should_print = (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % 250 == 0
            )

            if should_print and show_progress:
                elapsed = (
                    time.perf_counter()
                    - epoch_start
                )

                batches_per_second = (
                    batch_index
                    / max(elapsed, 1e-6)
                )

                remaining_batches = (
                    total_batches
                    - batch_index
                )

                eta_seconds = (
                    remaining_batches
                    / max(
                        batches_per_second,
                        1e-6,
                    )
                )

                percentage = (
                    100.0
                    * batch_index
                    / total_batches
                )

                print(
                    f"{phase:<10} | "
                    f"EPOCH {epoch:02d}/{total_epochs:02d} | "
                    f"BATCH {batch_index:,}/"
                    f"{total_batches:,} | "
                    f"{percentage:6.2f}% | "
                    f"LOSS {running_loss:.4f} | "
                    f"PPL {running_perplexity:.2f} | "
                    f"TOKENS {total_tokens:,} | "
                    f"ETA {format_time(eta_seconds)}",
                    flush=True,
                )

    if total_tokens == 0:
        raise RuntimeError(
            f"No non-padding target tokens found in {phase}."
        )

    average_loss = (
        total_loss
        / total_tokens
    )

    perplexity = calculate_perplexity(
        average_loss
    )

    elapsed_seconds = (
        time.perf_counter()
        - epoch_start
    )

    print(
        f"{phase} FINISHED | "
        f"EPOCH {epoch:02d}/{total_epochs:02d} | "
        f"LOSS {average_loss:.4f} | "
        f"PPL {perplexity:.2f} | "
        f"TIME {format_time(elapsed_seconds)}",
        flush=True,
    )

    return (
        average_loss,
        perplexity,
        elapsed_seconds,
        total_tokens,
    )

def save_checkpoint(
    checkpoint_path: Path,
    model: TransformerNMT,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    best_valid_loss: float,
    config: dict[str, Any],
) -> None:
    """
    Save complete model-training state.

    This allows training to resume with the model, optimizer,
    scheduler, epoch, and best validation loss restored.
    """
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "best_valid_loss": best_valid_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: TransformerNMT,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    current_config: dict[str, Any],
) -> tuple[int, float, list[dict[str, Any]]]:
    """
    Load a previous checkpoint.

    The checkpoint must use the same architecture and tokenizer
    configuration as the current run.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Resume checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print("=" * 88)
    print("LOADING CHECKPOINT FOR RESUME")
    print("=" * 88)
    print(f"CHECKPOINT: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint must contain a dictionary."
        )

    required_keys = {
        "epoch",
        "valid_loss",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    saved_config = checkpoint.get(
        "config"
    )

    if isinstance(saved_config, dict):
        saved_model_config = (
            saved_config.get("model")
            or saved_config
        )

        current_model_config = (
            current_config.get("model")
            or current_config
        )

        if isinstance(
            saved_model_config,
            dict,
        ):
            if saved_model_config != current_model_config:
                raise RuntimeError(
                    "Checkpoint model configuration does not "
                    "match the current command.\n\n"
                    f"SAVED CONFIG:\n{saved_model_config}\n\n"
                    f"CURRENT CONFIG:\n{current_model_config}"
                )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    last_epoch = int(
        checkpoint["epoch"]
    )

    history: list[dict[str, Any]] = []

    history_path = (
        checkpoint_path.parent
        / "training_history.json"
    )

    if history_path.is_file():
        try:
            loaded_history = json.loads(
                history_path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                loaded_history,
                list,
            ):
                history = [
                    item
                    for item in loaded_history
                    if isinstance(item, dict)
                ]

        except (
            OSError,
            json.JSONDecodeError,
        ):
            print(
                "WARNING: Existing training history "
                "could not be loaded."
            )

    history_losses = [
        float(item["valid_loss"])
        for item in history
        if "valid_loss" in item
    ]

    saved_best_loss = checkpoint.get(
        "best_valid_loss"
    )

    if saved_best_loss is not None:
        best_valid_loss = float(
            saved_best_loss
        )

    elif history_losses:
        best_valid_loss = min(
            history_losses
        )

    else:
        best_valid_loss = float(
            checkpoint["valid_loss"]
        )

    print("CHECKPOINT LOADED SUCCESSFULLY")
    print(
        f"LAST COMPLETED EPOCH : "
        f"{last_epoch}"
    )
    print(
        f"BEST VALIDATION LOSS : "
        f"{best_valid_loss:.4f}"
    )
    print(
        f"NEXT EPOCH           : "
        f"{last_epoch + 1}"
    )
    print("=" * 88)
    print()

    return (
        last_epoch,
        best_valid_loss,
        history,
    )


def write_history_json(
    history_path: Path,
    history: list[dict[str, Any]],
) -> None:
    """Write complete training history as JSON."""
    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_history_csv(
    history_path: Path,
    history: list[dict[str, Any]],
) -> None:
    """Write complete training history as CSV."""
    if not history:
        return

    fieldnames = [
        "epoch",
        "train_loss",
        "train_perplexity",
        "valid_loss",
        "valid_perplexity",
        "learning_rate",
        "train_time_seconds",
        "valid_time_seconds",
        "epoch_time_seconds",
        "elapsed_time_seconds",
        "eta_seconds",
        "best_valid_loss",
        "best_checkpoint_saved",
    ]

    with history_path.open(
        "w",
<<<<<<< HEAD
        newline="", 
=======
        newline="",
        encoding="utf-8",
>>>>>>> 51da30f (Prepare NMT project for deployment)
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(history)


def print_epoch_result(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_perplexity: float,
    valid_loss: float,
    valid_perplexity: float,
    learning_rate: float,
    train_time: float,
    valid_time: float,
    elapsed_time: float,
    eta_seconds: float,
    best_valid_loss: float,
    checkpoint_saved: bool,
) -> None:
    """Print a highly visible uppercase epoch summary."""
    print()
    print("#" * 88)
    print(
        f"### EPOCH {epoch:02d}/{total_epochs:02d} "
        "COMPLETED ###"
    )
    print("#" * 88)
    print(
        f"TRAIN LOSS            : "
        f"{train_loss:.4f}"
    )
    print(
        f"TRAIN PERPLEXITY      : "
        f"{train_perplexity:.2f}"
    )
    print(
        f"VALIDATION LOSS       : "
        f"{valid_loss:.4f}"
    )
    print(
        f"VALIDATION PERPLEXITY : "
        f"{valid_perplexity:.2f}"
    )
    print(
        f"LEARNING RATE         : "
        f"{learning_rate:.6e}"
    )
    print(
        f"TRAINING TIME         : "
        f"{format_time(train_time)}"
    )
    print(
        f"VALIDATION TIME       : "
        f"{format_time(valid_time)}"
    )
    print(
        f"TOTAL EPOCH TIME      : "
        f"{format_time(train_time + valid_time)}"
    )
    print(
        f"SESSION ELAPSED       : "
        f"{format_time(elapsed_time)}"
    )
    print(
        f"ESTIMATED TIME LEFT   : "
        f"{format_time(eta_seconds)}"
    )
    print(
        f"BEST VALIDATION LOSS  : "
        f"{best_valid_loss:.4f}"
    )
    print(
        f"BEST CHECKPOINT       : "
        f"{'NEW BEST SAVED' if checkpoint_saved else 'NOT IMPROVED'}"
    )
    print("#" * 88)
    print()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""
    if args.vocab_size <= 4:
        raise ValueError(
            "--vocab-size must be greater than 4."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be positive."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay cannot be negative."
        )

    if args.max_source_length <= 0:
        raise ValueError(
            "--max-source-length must be positive."
        )

    if args.max_target_length < 3:
        raise ValueError(
            "--max-target-length must be at least 3."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(
            "--dropout must be in the range [0, 1)."
        )

    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError(
            "--label-smoothing must be in the range [0, 1)."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train an Amharic-to-Afaan-Oromo "
            "Transformer NMT model."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--feedforward-dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-source-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    parser.add_argument(
        "--max-target-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:0.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from checkpoints/"
            "last_transformer.pt."
        ),
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable live batch progress bars "
            "but keep epoch summaries."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    set_seed(args.seed)

    device = select_device(args.device)

    project_root = args.project_root.resolve()

    tokenized_dir = (
        project_root
        / "tokenizer"
        / "tokenized"
    )

    checkpoint_dir = (
        project_root
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_checkpoint_path = (
        checkpoint_dir
        / "last_transformer.pt"
    )

    best_checkpoint_path = (
        checkpoint_dir
        / "best_transformer.pt"
    )

    history_json_path = (
        checkpoint_dir
        / "training_history.json"
    )

    history_csv_path = (
        checkpoint_dir
        / "training_history.csv"
    )

    print()
    print("=" * 88)
    print("AMHARIC → AFAAN OROMO TRANSFORMER TRAINING")
    print("=" * 88)
    print(f"DEVICE               : {device}")
    print(f"PROJECT ROOT         : {project_root}")
    print(f"TOKENIZED DATA       : {tokenized_dir}")
    print(f"CHECKPOINT DIRECTORY : {checkpoint_dir}")
    print("=" * 88)
    print()

    train_loader, valid_loader, _ = (
        create_dataloaders(
            tokenized_dir=tokenized_dir,
            batch_size=args.batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            num_workers=args.num_workers,
        )
    )

    model_config = {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_encoder_layers": (
            args.encoder_layers
        ),
        "num_decoder_layers": (
            args.decoder_layers
        ),
        "feedforward_dim": (
            args.feedforward_dim
        ),
        "dropout": args.dropout,
        "max_sequence_length": max(
            args.max_source_length,
            args.max_target_length,
        ),
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
    }

    model = TransformerNMT(
        **model_config
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"TRAINABLE PARAMETERS  : "
        f"{format_number(parameter_count)}"
    )

    print(
        f"TRAINING BATCHES      : "
        f"{format_number(len(train_loader))}"
    )

    print(
        f"VALIDATION BATCHES    : "
        f"{format_number(len(valid_loader))}"
    )

    print()

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = create_grad_scaler(device)

    training_config = {
        "task": (
            "Amharic to Afaan Oromo "
            "Neural Machine Translation"
        ),
        "dataset": {
            "train_pairs": len(
                train_loader.dataset
            ),
            "valid_pairs": len(
                valid_loader.dataset
            ),
        },
        "model": model_config,
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": args.weight_decay,
            "label_smoothing": (
                args.label_smoothing
            ),
            "gradient_clip_norm": (
                DEFAULT_GRAD_CLIP
            ),
            "seed": args.seed,
            "device": str(device),
            "amp_enabled": scaler is not None,
        },
    }

    history: list[dict[str, Any]] = []
    best_valid_loss = float("inf")
    start_epoch = 1

    if args.resume:
        (
            last_epoch,
            best_valid_loss,
            history,
        ) = load_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            current_config=training_config,
        )

        start_epoch = last_epoch + 1

        if start_epoch > args.epochs:
            print()
            print("=" * 88)
            print("TRAINING IS ALREADY COMPLETE")
            print("=" * 88)
            print(
                f"LAST COMPLETED EPOCH : "
                f"{last_epoch}"
            )
            print(
                f"TARGET EPOCHS        : "
                f"{args.epochs}"
            )
            print(
                f"BEST VALIDATION LOSS : "
                f"{best_valid_loss:.4f}"
            )
            print("=" * 88)
            return

    print()
    print("=" * 88)

    if args.resume:
        print(
            f"RESUMING TRAINING FROM "
            f"EPOCH {start_epoch:02d}/"
            f"{args.epochs:02d}"
        )
    else:
        print(
            f"STARTING NEW TRAINING RUN: "
            f"{args.epochs} EPOCHS"
        )

    print("=" * 88)
    print()

    session_start = time.perf_counter()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        train_loss, train_perplexity, train_time, _ = (
            run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                grad_clip_norm=DEFAULT_GRAD_CLIP,
                epoch=epoch,
                total_epochs=args.epochs,
                phase="TRAINING ",
                show_progress=not args.no_progress,
            )
        )

        valid_loss, valid_perplexity, valid_time, _ = (
            run_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                grad_clip_norm=DEFAULT_GRAD_CLIP,
                epoch=epoch,
                total_epochs=args.epochs,
                phase="VALIDATION",
                show_progress=not args.no_progress,
            )
        )

        scheduler.step(valid_loss)

        learning_rate = optimizer.param_groups[0]["lr"]

        elapsed_time = (
            time.perf_counter()
            - session_start
        )

        completed_epochs = (
            epoch - start_epoch + 1
        )

        remaining_epochs = (
            args.epochs - epoch
        )

        average_epoch_time = (
            elapsed_time
            / max(completed_epochs, 1)
        )

        eta_seconds = (
            average_epoch_time
            * remaining_epochs
        )

        checkpoint_improved = (
            valid_loss < best_valid_loss
        )

        if checkpoint_improved:
            best_valid_loss = valid_loss

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(
                train_loss,
                6,
            ),
            "train_perplexity": round(
                train_perplexity,
                4,
            ),
            "valid_loss": round(
                valid_loss,
                6,
            ),
            "valid_perplexity": round(
                valid_perplexity,
                4,
            ),
            "learning_rate": learning_rate,
            "train_time_seconds": round(
                train_time,
                2,
            ),
            "valid_time_seconds": round(
                valid_time,
                2,
            ),
            "epoch_time_seconds": round(
                train_time + valid_time,
                2,
            ),
            "elapsed_time_seconds": round(
                elapsed_time,
                2,
            ),
            "eta_seconds": round(
                eta_seconds,
                2,
            ),
            "best_valid_loss": round(
                best_valid_loss,
                6,
            ),
            "best_checkpoint_saved": (
                checkpoint_improved
            ),
        }

        history = [
            item
            for item in history
            if item.get("epoch") != epoch
        ]

        history.append(epoch_result)

        history.sort(
            key=lambda item: item.get(
                "epoch",
                0,
            )
        )

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            best_valid_loss=best_valid_loss,
            config=training_config,
        )

        if checkpoint_improved:
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                best_valid_loss=best_valid_loss,
                config=training_config,
            )

        write_history_json(
            history_path=history_json_path,
            history=history,
        )

        write_history_csv(
            history_path=history_csv_path,
            history=history,
        )

        print_epoch_result(
            epoch=epoch,
            total_epochs=args.epochs,
            train_loss=train_loss,
            train_perplexity=train_perplexity,
            valid_loss=valid_loss,
            valid_perplexity=valid_perplexity,
            learning_rate=learning_rate,
            train_time=train_time,
            valid_time=valid_time,
            elapsed_time=elapsed_time,
            eta_seconds=eta_seconds,
            best_valid_loss=best_valid_loss,
            checkpoint_saved=checkpoint_improved,
        )

    total_time = (
        time.perf_counter()
        - session_start
    )

    print()
    print("=" * 88)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 88)
    print(
        f"TOTAL TRAINING TIME   : "
        f"{format_time(total_time)}"
    )
    print(
        f"BEST VALIDATION LOSS  : "
        f"{best_valid_loss:.4f}"
    )
    print(
        f"BEST CHECKPOINT       : "
        f"{best_checkpoint_path}"
    )
    print(
        f"LAST CHECKPOINT       : "
        f"{last_checkpoint_path}"
    )
    print(
        f"JSON HISTORY          : "
        f"{history_json_path}"
    )
    print(
        f"CSV HISTORY           : "
        f"{history_csv_path}"
    )
    print("=" * 88)
    print()


if __name__ == "__main__":
    main()