from __future__ import annotations
import argparse
from pathlib import Path
import sentencepiece as spm
import torch
from transformer import TransformerNMT
import warnings

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage",
    category=UserWarning,
)
 
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(name)


    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    return device


def load_sentencepiece(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SentencePiece model not found: {model_path}"
        )

    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    return processor


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> TransformerNMT:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model configuration."
        )
    model = TransformerNMT(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def translate(
    text: str,
    processor,
    model: TransformerNMT,
    device: torch.device,
    max_source_length: int = 256,
    max_target_length: int = 256,
) -> str:
    source_ids = processor.encode(
        text,
        out_type=int,
    )

    source_ids = source_ids[:max_source_length]

    if not source_ids:
        raise ValueError("Input text produced no source tokens.")

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=device,
    )

    source_padding_mask = source.eq(PAD_ID)

    generated = model.greedy_decode(
        source=source,
        source_padding_mask=source_padding_mask,
        max_length=max_target_length,
    )

    output_ids = generated[0].tolist()

    if BOS_ID in output_ids:
        output_ids = output_ids[
            output_ids.index(BOS_ID) + 1:
        ]

    if EOS_ID in output_ids:
        output_ids = output_ids[
            :output_ids.index(EOS_ID)
        ]

    output_ids = [
        token_id
        for token_id in output_ids
        if token_id not in {PAD_ID, BOS_ID, EOS_ID}
    ]

    return processor.decode(output_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Amharic text into Afaan Oromo."
    )

    project_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "checkpoints" / "best_transformer.pt",
    )

    parser.add_argument(
        "--sp-model",
        type=Path,
        default=project_root
        / "tokenizer"
        / "translator_sp.model",
    )

    parser.add_argument("--text", type=str)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    processor = load_sentencepiece(args.sp_model)
    model = load_model(args.checkpoint, device)

    print("\n" + "=" * 70)
    print("AMHARIC → AFAAN OROMO TRANSLATOR")
    print("=" * 70)
    print("Type an Amharic sentence to translate.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    # If --text was provided, translate it once and exit
    if args.text:
        translation = translate(
            text=args.text,
            processor=processor,
            model=model,
            device=device,
            max_target_length=args.max_length,
        )

        print(f"Amharic: {args.text}")
        print(f"Afaan Oromo: {translation}")
        return

    # Interactive translation loop
    while True:
        try:
            text = input("\nAmharic: ").strip()

            if text.lower() in {"exit", "quit"}:
                print("Exiting translator. Goodbye!")
                break

            if not text:
                print("Please enter an Amharic sentence.")
                continue

            translation = translate(
                text=text,
                processor=processor,
                model=model,
                device=device,
                max_target_length=args.max_length,
            )

            print(f"Afaan Oromo: {translation}")

        except KeyboardInterrupt:
            print("\nExiting translator. Goodbye!")
            break

        except Exception as e:
            print(f"Translation error: {e}")


if __name__ == "__main__":
    main()