from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


@dataclass(frozen=True)
class Batch:
    source: torch.Tensor
    target: torch.Tensor
    source_padding_mask: torch.Tensor
    target_padding_mask: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            source=self.source.to(device, non_blocking=True),
            target=self.target.to(device, non_blocking=True),
            source_padding_mask=self.source_padding_mask.to(
                device, non_blocking=True
            ),
            target_padding_mask=self.target_padding_mask.to(
                device, non_blocking=True
            ),
        )


class ParallelIdsDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for aligned source and target SentencePiece ID files."""

    def __init__(
        self,
        source_path: Path,
        target_path: Path,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ) -> None:
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        print(f"📂 Loading dataset files from: {self.source_path.parent}")
        source_sequences = self._read_sequences(self.source_path)
        target_sequences = self._read_sequences(self.target_path)

        if len(source_sequences) != len(target_sequences):
            raise ValueError(
                "Source and target line counts do not match: "
                f"{self.source_path}={len(source_sequences)}, "
                f"{self.target_path}={len(target_sequences)}"
            )

        self.examples: list[tuple[list[int], list[int]]] = []

        for source_ids, target_ids in zip(
            source_sequences,
            target_sequences,
            strict=True,
        ):
            if not source_ids or not target_ids:
                continue

            source_ids = source_ids[:max_source_length]

            # Reserve two positions for BOS and EOS.
            target_ids = target_ids[
                : max(1, max_target_length - 2)
            ]

            if not source_ids or not target_ids:
                continue

            source = source_ids
            target = [BOS_ID, *target_ids, EOS_ID]

            self.examples.append((source, target))

        if not self.examples:
            raise ValueError(
                f"No usable examples found in {self.source_path}"
            )
            
        print(f"✓ Loaded {len(self.examples):,} valid sentence pairs from {self.source_path.name}")

    @staticmethod
    def _read_sequences(path: Path) -> list[list[int]]:
        if not path.is_file():
            raise FileNotFoundError(f"ID file not found: {path}")

        sequences: list[list[int]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()

                if not text:
                    sequences.append([])
                    continue

                try:
                    sequence = [int(value) for value in text.split()]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid integer in {path}, line {line_number}"
                    ) from exc

                sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids, target_ids = self.examples[index]

        return (
            torch.tensor(source_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def collate_batch(
    examples: Sequence[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = PAD_ID,
) -> Batch:
    if not examples:
        raise ValueError("Cannot collate an empty batch.")

    sources, targets = zip(*examples)

    source = pad_sequence(
        sources,
        batch_first=True,
        padding_value=pad_id,
    )

    target = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_id,
    )

    source_padding_mask = source.eq(pad_id)
    target_padding_mask = target.eq(pad_id)

    return Batch(
        source=source,
        target=target,
        source_padding_mask=source_padding_mask,
        target_padding_mask=target_padding_mask,
    )


def create_dataloaders(
    tokenized_dir: Path,
    batch_size: int = 32,
    max_source_length: int = 256,
    max_target_length: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader[Batch], DataLoader[Batch], DataLoader[Batch]]:
    tokenized_dir = Path(tokenized_dir)
    print("\n--- Initializing Dataset & Dataloaders ---")

    train_dataset = ParallelIdsDataset(
        tokenized_dir / "amharic.train.ids.txt",
        tokenized_dir / "afan_oromo.train.ids.txt",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    valid_dataset = ParallelIdsDataset(
        tokenized_dir / "amharic.valid.ids.txt",
        tokenized_dir / "afan_oromo.valid.ids.txt",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    test_dataset = ParallelIdsDataset(
        tokenized_dir / "amharic.test.ids.txt",
        tokenized_dir / "afan_oromo.test.ids.txt",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_batch,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    print("✓ Dataloaders successfully created and ready!\n")
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    import sys
    # Quick sanity check block when dataset.py is run directly
    tokenized_path = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenized"
    if tokenized_path.exists():
        print("Testing dataset loader execution...")
        train_l, valid_l, test_l = create_dataloaders(tokenized_path, batch_size=16)
        print(f"Total training batches: {len(train_l):,}")
    else:
        print(f"Tokenized directory not found at: {tokenized_path}. Run tokenizer.py first.")