from __future__ import annotations

import argparse
from pathlib import Path

import sacrebleu
import torch

from dataset import create_dataloaders
from inference import load_model, load_sentencepiece, select_device


def remove_special_tokens(
    token_ids: list[int],
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
) -> list[int]:
    result: list[int] = []

    for token_id in token_ids:
        if token_id == eos_id:
            break

        if token_id not in {pad_id, bos_id}:
            result.append(token_id)

    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Calculate corpus BLEU on the test set."
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

    return parser.parse_args()


@torch.no_grad()
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

    processor = load_sentencepiece(args.sp_model)
    model = load_model(args.checkpoint, device)

    predictions: list[str] = []
    references: list[str] = []

    for batch in test_loader:
        batch = batch.to(device)

        generated = model.greedy_decode(
            source=batch.source,
            source_padding_mask=batch.source_padding_mask,
            max_length=config["max_sequence_length"],
        )

        for generated_ids, target_ids in zip(
            generated.tolist(),
            batch.target.tolist(),
            strict=True,
        ):
            generated_ids = remove_special_tokens(generated_ids)
            target_ids = remove_special_tokens(target_ids)

            predictions.append(processor.decode(generated_ids))
            references.append(processor.decode(target_ids))

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    print(f"BLEU: {bleu.score:.2f}")
    print(f"Evaluated translations: {len(predictions)}")


if __name__ == "__main__":
    main()