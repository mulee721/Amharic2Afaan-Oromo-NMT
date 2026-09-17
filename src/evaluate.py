from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import sacrebleu
import sentencepiece as spm
import torch
from torch import nn

from dataset import Batch, create_dataloaders
from inference import load_model, select_device
from transformer import TransformerNMT

DEFAULT_PAD_ID = 0
DEFAULT_UNK_ID = 1
DEFAULT_BOS_ID = 2
DEFAULT_EOS_ID = 3


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Amharic-to-Afaan-Oromo "
            "Transformer using loss, perplexity, "
            "BLEU, and chrF."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            project_root
            / "checkpoints"
            / "best_transformer.pt"
        ),
    )

    parser.add_argument(
        "--sp-model",
        type=Path,
        default=(
            project_root
            / "tokenizer"
            / "translator_sp.model"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
    )

    return parser.parse_args()


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load and validate a checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "The checkpoint must contain a dictionary."
        )

    if "config" not in checkpoint:
        raise RuntimeError(
            "The checkpoint does not contain 'config'."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "The checkpoint does not contain "
            "'model_state_dict'."
        )

    return checkpoint


def get_model_config(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """
    Read model configuration.

    Supports both:

        checkpoint["config"]["model"]

    and older:

        checkpoint["config"]
    """
    config = checkpoint["config"]

    if not isinstance(config, dict):
        raise TypeError(
            "checkpoint['config'] must be a dictionary."
        )

    model_config = config.get("model", config)

    if not isinstance(model_config, dict):
        raise TypeError(
            "Model configuration must be a dictionary."
        )

    if "max_sequence_length" not in model_config:
        raise KeyError(
            "Missing 'max_sequence_length'. Available "
            f"keys: {list(model_config.keys())}"
        )

    if "vocab_size" not in model_config:
        raise KeyError(
            "Missing 'vocab_size' from model configuration."
        )

    return model_config


def load_sentencepiece(
    model_path: Path,
    expected_vocab_size: int,
) -> spm.SentencePieceProcessor:
    """Load and validate the SentencePiece model."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SentencePiece model not found:\n{model_path}"
        )

    processor = spm.SentencePieceProcessor()

    if not processor.load(str(model_path)):
        raise RuntimeError(
            f"Could not load SentencePiece model:\n{model_path}"
        )

    actual_vocab_size = processor.get_piece_size()

    if actual_vocab_size != expected_vocab_size:
        raise RuntimeError(
            "Vocabulary-size mismatch:\n"
            f"Checkpoint vocabulary: {expected_vocab_size}\n"
            f"SentencePiece vocabulary: {actual_vocab_size}"
        )

    return processor


def remove_special_tokens(
    token_ids: list[int],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    """
    Remove PAD and BOS tokens and stop at EOS.

    UNK is intentionally kept because it is a real
    vocabulary token.
    """
    cleaned_ids: list[int] = []

    for token_id in token_ids:
        if token_id == eos_id:
            break

        if token_id in {pad_id, bos_id}:
            continue

        cleaned_ids.append(token_id)

    return cleaned_ids


def calculate_perplexity(loss: float) -> float:
    """Convert loss into perplexity safely."""
    return math.exp(min(loss, 20.0))


@torch.inference_mode()
def evaluate_loss(
    model: TransformerNMT,
    loader,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int,
) -> tuple[float, float, int]:
    """Calculate test loss, perplexity, and valid tokens."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch: Batch = batch.to(device)

        target_input = batch.target[:, :-1]
        target_labels = batch.target[:, 1:]

        target_padding_mask = target_input.eq(
            pad_id
        )

        logits = model(
            source=batch.source,
            target_input=target_input,
            source_padding_mask=batch.source_padding_mask,
            target_padding_mask=target_padding_mask,
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            target_labels.reshape(-1),
        )

        token_count = int(
            target_labels.ne(pad_id).sum().item()
        )

        total_loss += (
            loss.item() * token_count
        )

        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError(
            "No valid non-padding tokens found."
        )

    average_loss = total_loss / total_tokens
    perplexity = calculate_perplexity(
        average_loss
    )

    return (
        average_loss,
        perplexity,
        total_tokens,
    )


@torch.inference_mode()
def generate_translations_with_sources(
    model: TransformerNMT,
    loader,
    processor: spm.SentencePieceProcessor,
    device: torch.device,
    max_length: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Generate decoded predictions, references, and sources.

    Same as generate_translations, but also returns source sentences.
    """
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    sources: list[str] = []

    for batch in loader:
        batch: Batch = batch.to(device)

        generated = model.greedy_decode(
            source=batch.source,
            source_padding_mask=batch.source_padding_mask,
            max_length=max_length,
        )

        # Decode source sentences as well
        source_list = batch.source.tolist()
        target_list = batch.target.tolist()
        generated_list = generated.tolist()

        for src_ids, gen_ids, tgt_ids in zip(
            source_list,
            generated_list,
            target_list,
            strict=True,
        ):
            # For sources, you may also want to remove special tokens
            clean_src_ids = remove_special_tokens(
                src_ids,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )

            clean_gen_ids = remove_special_tokens(
                gen_ids,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )

            clean_tgt_ids = remove_special_tokens(
                tgt_ids,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
            )

            sources.append(processor.decode(clean_src_ids))
            predictions.append(processor.decode(clean_gen_ids))
            references.append(processor.decode(clean_tgt_ids))

    return sources, predictions, references


def print_examples(
    predictions: list[str],
    references: list[str],
    number_of_examples: int,
) -> None:
    """Print a few model predictions and references."""
    if number_of_examples <= 0:
        return

    count = min(
        number_of_examples,
        len(predictions),
    )

    print()
    print("=" * 100)
    print("SAMPLE TEST TRANSLATIONS")
    print("=" * 100)

    for index in range(count):
        print()
        print(f"EXAMPLE {index + 1}")
        print(f"REFERENCE  : {references[index]}")
        print(f"PREDICTION : {predictions[index]}")

    print()
    print("=" * 100)


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.show_examples < 0:
        raise ValueError(
            "--show-examples cannot be negative."
        )

    device = select_device(args.device)

    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model_config = get_model_config(
        checkpoint
    )

    vocab_size = int(
        model_config["vocab_size"]
    )

    max_sequence_length = int(
        model_config["max_sequence_length"]
    )

    pad_id = int(
        model_config.get(
            "pad_id",
            DEFAULT_PAD_ID,
        )
    )

    bos_id = int(
        model_config.get(
            "bos_id",
            DEFAULT_BOS_ID,
        )
    )

    eos_id = int(
        model_config.get(
            "eos_id",
            DEFAULT_EOS_ID,
        )
    )

    tokenized_dir = (
        args.project_root
        / "tokenizer"
        / "tokenized"
    )

    _, _, test_loader = create_dataloaders(
        tokenized_dir=tokenized_dir,
        batch_size=args.batch_size,
        max_source_length=max_sequence_length,
        max_target_length=max_sequence_length,
        num_workers=0,
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    processor = load_sentencepiece(
        model_path=args.sp_model,
        expected_vocab_size=vocab_size,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id,
    )

    test_loss, test_perplexity, valid_tokens = (
        evaluate_loss(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            pad_id=pad_id,
        )
    )

    # UPDATED: get sources as well
    sources, predictions, references = (
        generate_translations_with_sources(
            model=model,
            loader=test_loader,
            processor=processor,
            device=device,
            max_length=max_sequence_length,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
    )

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    )

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
    )

    print()
    print("=" * 100)
    print("MACHINE TRANSLATION EVALUATION RESULTS")
    print("=" * 100)
    print(
        f"CHECKPOINT       : {args.checkpoint}"
    )
    print(
        f"CHECKPOINT EPOCH : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(f"DEVICE           : {device}")
    print(
        f"TEST PAIRS       : "
        f"{len(test_loader.dataset):,}"
    )
    print(
        f"VALID TOKENS     : "
        f"{valid_tokens:,}"
    )
    print()
    print(f"TEST LOSS        : {test_loss:.4f}")
    print(
        f"TEST PERPLEXITY  : "
        f"{test_perplexity:.2f}"
    )
    print()
    print(f"BLEU             : {bleu.score:.2f}")
    print(f"chrF             : {chrf.score:.2f}")
    print("=" * 100)

    print_examples(
        predictions=predictions,
        references=references,
        number_of_examples=args.show_examples,
    )
if __name__ == "__main__":
    main()