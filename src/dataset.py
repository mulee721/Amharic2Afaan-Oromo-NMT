from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0
UNK_ID = 1
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
                device,
                non_blocking=True,
            ),
            target_padding_mask=self.target_padding_mask.to(
                device,
                non_blocking=True,
            ),
        )

    def pin_memory(self) -> "Batch":
        return Batch(
            source=self.source.pin_memory(),
            target=self.target.pin_memory(),
            source_padding_mask=self.source_padding_mask.pin_memory(),
            target_padding_mask=self.target_padding_mask.pin_memory(),
        )


class ParallelIdsDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for aligned source and target SentencePiece ID files."""

    def __init__(
        self,
        source_path: Path,
        target_path: Path,
        max_source_length: int = 64,
        max_target_length: int = 64,
    ) -> None:
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        if max_source_length <= 0:
            raise ValueError("max_source_length must be positive.")

        if max_target_length < 3:
            raise ValueError(
                "max_target_length must be at least 3 "
                "for BOS, one token, and EOS."
            )

        print(
            f"📂 Loading source data: {self.source_path.name}"
        )

        source_sequences = self._read_sequences(
            self.source_path
        )

        print(
            f"📂 Loading target data: {self.target_path.name}"
        )

        target_sequences = self._read_sequences(
            self.target_path
        )

        if len(source_sequences) != len(target_sequences):
            raise ValueError(
                "Source and target line counts do not match: "
                f"{self.source_path}={len(source_sequences)}, "
                f"{self.target_path}={len(target_sequences)}"
            )

        self.examples: list[tuple[list[int], list[int]]] = []
        skipped_empty_pairs = 0

        for line_number, (source_ids, target_ids) in enumerate(
            zip(
                source_sequences,
                target_sequences,
                strict=True,
            ),
            start=1,
        ):
            if not source_ids or not target_ids:
                skipped_empty_pairs += 1
                continue

            if len(source_ids) > max_source_length:
                raise ValueError(
                    f"Source sequence is too long at line "
                    f"{line_number} in {self.source_path.name}: "
                    f"{len(source_ids)} > {max_source_length}"
                )

            target_length_with_specials = len(target_ids) + 2

            if target_length_with_specials > max_target_length:
                raise ValueError(
                    f"Target sequence is too long at line "
                    f"{line_number} in {self.target_path.name}: "
                    f"{target_length_with_specials} > "
                    f"{max_target_length} after adding BOS/EOS"
                )

            target = [BOS_ID, *target_ids, EOS_ID]

            self.examples.append(
                (source_ids, target)
            )

        if not self.examples:
            raise ValueError(
                f"No usable examples found in {self.source_path}"
            )

        print(
            f"✓ Loaded {len(self.examples):,} aligned sentence pairs "
            f"from {self.source_path.name}"
        )

        if skipped_empty_pairs:
            print(
                f"⚠️ Skipped empty pairs: "
                f"{skipped_empty_pairs:,}"
            )

    @staticmethod
    def _read_sequences(path: Path) -> list[list[int]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"ID file not found: {path}"
            )

        sequences: list[list[int]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                text = line.strip()

                if not text:
                    sequences.append([])
                    continue

                try:
                    sequence = [
                        int(value)
                        for value in text.split()
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid token ID in {path}, "
                        f"line {line_number}"
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
    """Dynamically pad source and target sequences in one batch."""

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
    batch_size: int = 16,
    max_source_length: int = 64,
    max_target_length: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader[Batch], DataLoader[Batch], DataLoader[Batch]]:
    """Create train, validation, and test DataLoaders."""

    tokenized_dir = Path(tokenized_dir)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    print("\n--- Initializing datasets and dataloaders ---")

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

    print("✓ Dataloaders successfully created.\n")

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    tokenized_path = (
        Path(__file__).resolve().parent.parent
        / "tokenizer"
        / "tokenized"
    )

    if not tokenized_path.exists():
        raise FileNotFoundError(
            f"Tokenized directory not found: {tokenized_path}\n"
            "Run the tokenizer pipeline first."
        )

    print("Testing dataset loader execution...")

    train_loader, valid_loader, test_loader = create_dataloaders(
        tokenized_dir=tokenized_path,
        batch_size=16,
        max_source_length=64,
        max_target_length=64,
    )

    batch = next(iter(train_loader))
    print(f"Training pairs: {len(train_loader.dataset):,}")
    print(f"Validation pairs: {len(valid_loader.dataset):,}")
    print(f"Test pairs: {len(test_loader.dataset):,}")
    print(f"Training batches: {len(train_loader):,}")
    print(f"Source batch shape: {tuple(batch.source.shape)}")
    print(f"Target batch shape: {tuple(batch.target.shape)}")
    print("✓ Dataset sanity check passed.")