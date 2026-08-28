from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from dataset import Batch, create_dataloaders
from inference import load_model, load_sentencepiece, select_device
from transformer import TransformerNMT


PAD_ID = 0


@torch.no_grad()
def evaluate_loss(
    model: TransformerNMT,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch: Batch = batch.to(device)

        target_input = batch.target[:, :-1]
        target_labels = batch.target[:, 1:]

        logits = model(
            source=batch.source,
            target_input=target_input,
            source_padding_mask=batch.source_padding_mask,
            target_padding_mask=target_input.eq(PAD_ID),
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_labels.reshape(-1),
        )

        token_count = target_labels.ne(PAD_ID).sum().item()

        total_loss += loss.item() * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("No valid test tokens found.")

    loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(loss)).item()

    return loss, perplexity


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Evaluate the Amharic-to-Oromo Transformer."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "checkpoints"
        / "best_transformer.pt",
    )

    parser.add_argument(
        "--sp-model",
        type=Path,
        default=project_root
        / "tokenizer"
        / "translator_sp.model",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-examples", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    _, _, test_loader = create_dataloaders(
        tokenized_dir=args.project_root
        / "tokenizer"
        / "tokenized",
        batch_size=args.batch_size,
        max_source_length=config["max_sequence_length"],
        max_target_length=config["max_sequence_length"],
    )

    model = load_model(args.checkpoint, device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
    )

    loss, perplexity = evaluate_loss(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test loss: {loss:.4f}")
    print(f"Test perplexity: {perplexity:.2f}")

    if args.show_examples > 0:
        processor = load_sentencepiece(args.sp_model)

        batch = next(iter(test_loader)).to(device)
        predictions = model.greedy_decode(
            source=batch.source[:args.show_examples],
            source_padding_mask=batch.source_padding_mask[
                :args.show_examples
            ],
            max_length=config["max_sequence_length"],
        )

        for index, prediction in enumerate(predictions):
            source_text = processor.decode(
                batch.source[index].tolist()
            )

            target_text = processor.decode(
                batch.target[index].tolist()
            )

            predicted_ids = prediction.tolist()
            predicted_ids = [
                token_id
                for token_id in predicted_ids
                if token_id not in {0, 2, 3}
            ]

            prediction_text = processor.decode(predicted_ids)

            print("\nExample", index + 1)
            print("Source:    ", source_text)
            print("Reference: ", target_text)
            print("Prediction: ", prediction_text)


if __name__ == "__main__":
    main()