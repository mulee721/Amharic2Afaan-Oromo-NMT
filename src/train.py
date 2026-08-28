from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import Batch, create_dataloaders
from transformer import TransformerNMT


PAD_ID = 0
DEFAULT_VOCAB_SIZE = 10_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    return device


def run_epoch(
    model: TransformerNMT,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch: Batch = batch.to(device)

        target_input = batch.target[:, :-1]
        target_labels = batch.target[:, 1:]
        target_padding_mask = target_input.eq(PAD_ID)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                source=batch.source,
                target_input=target_input,
                source_padding_mask=batch.source_padding_mask,
                target_padding_mask=target_padding_mask,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_labels.reshape(-1),
            )

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
                optimizer.step()

        token_count = target_labels.ne(PAD_ID).sum().item()

        total_loss += loss.item() * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("No non-padding target tokens were found.")

    average_loss = total_loss / total_tokens
    perplexity = math.exp(min(average_loss, 20.0))

    return average_loss, perplexity


def save_checkpoint(
    path: Path,
    model: TransformerNMT,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    valid_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "valid_loss": valid_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Amharic-to-Afaan-Oromo Transformer."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)
    project_root = args.project_root.resolve()

    tokenized_dir = project_root / "tokenizer" / "tokenized"
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Tokenized data: {tokenized_dir}")

    train_loader, valid_loader, _ = create_dataloaders(
        tokenized_dir=tokenized_dir,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
    )

    model_config = {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "max_sequence_length": max(
            args.max_source_length,
            args.max_target_length,
        ),
        "pad_id": PAD_ID,
        "bos_id": 2,
        "eos_id": 3,
    }

    model = TransformerNMT(**model_config).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    training_config = {
        **model_config,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
    }

    best_valid_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        with torch.no_grad():
            valid_loss, valid_ppl = run_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
            )

        scheduler.step(valid_loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": train_ppl,
            "valid_loss": valid_loss,
            "valid_perplexity": valid_ppl,
            "learning_rate": learning_rate,
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | "
            f"valid loss {valid_loss:.4f} | "
            f"valid ppl {valid_ppl:.2f} | "
            f"lr {learning_rate:.2e}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            save_checkpoint(
                path=checkpoint_dir / "best_transformer.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                valid_loss=valid_loss,
                config=training_config,
            )

            print("  Saved best_transformer.pt")

    save_checkpoint(
        path=checkpoint_dir / "last_transformer.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        valid_loss=history[-1]["valid_loss"],
        config=training_config,
    )

    (checkpoint_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("Training completed.")
    print(f"Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()