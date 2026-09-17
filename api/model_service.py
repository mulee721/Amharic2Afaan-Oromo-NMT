from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

# Add project root/src to sys.path to import internal modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inference import load_model, load_sentencepiece, select_device

logger = logging.getLogger(__name__)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class TranslationError(Exception):
    """Raised for recoverable, user-facing translation failures.

    The API layer catches this specifically and turns it into a clean
    400-level response instead of leaking a stack trace.
    """


class TranslationService:
    def __init__(
        self,
        checkpoint_path: Path,
        sp_model_path: Path,
        device: str = "auto",
    ) -> None:
        self.device = select_device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        config = checkpoint.get("config")
        if config is None:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is missing a 'config' entry."
            )
        model_config = config.get("model", config)

        if "max_sequence_length" not in model_config:
            raise ValueError(
                "Model config is missing required key 'max_sequence_length'."
            )
        self.max_sequence_length = int(model_config["max_sequence_length"])

        self.pad_id = int(model_config.get("pad_id", PAD_ID))
        self.unk_id = int(model_config.get("unk_id", UNK_ID))
        self.bos_id = int(model_config.get("bos_id", BOS_ID))
        self.eos_id = int(model_config.get("eos_id", EOS_ID))

        # Config/metadata already extracted above -- drop the raw checkpoint
        # dict so we're not holding a second copy of the weights in memory
        # alongside whatever load_model() below loads internally.
        del checkpoint

        self.model = load_model(checkpoint_path, self.device)
        self.processor = load_sentencepiece(sp_model_path)

        self.model.eval()

    def remove_special_tokens(self, token_ids: list[int]) -> list[int]:
        cleaned: list[int] = []

        for token_id in token_ids:
            if token_id == self.eos_id:
                break

            if token_id in {self.pad_id, self.bos_id}:
                continue

            cleaned.append(token_id)

        return cleaned

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        if not text or not text.strip():
            raise TranslationError("Input text must not be empty.")

        source_ids = self.processor.encode_as_ids(text)

        if not source_ids:
            raise TranslationError(
                "Input text could not be tokenized into any tokens."
            )

        # Guard against overflowing the positional encoding / model's
        # supported length instead of letting it raise deep inside forward().
        if len(source_ids) > self.max_sequence_length:
            logger.warning(
                "Input tokenized to %d tokens; truncating to max_sequence_length=%d.",
                len(source_ids),
                self.max_sequence_length,
            )
            source_ids = source_ids[: self.max_sequence_length]

        source_tensor = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=self.device,
        )

        source_padding_mask = source_tensor.eq(self.pad_id)

        try:
            generated = self.model.greedy_decode(
                source=source_tensor,
                source_padding_mask=source_padding_mask,
                max_length=self.max_sequence_length,
            )
        except Exception as exc:
            raise TranslationError(
                f"Model failed to generate a translation: {exc}"
            ) from exc

        if generated is None or generated.numel() == 0:
            raise TranslationError("Model returned an empty output.")

        generated_ids = generated[0].tolist()
        cleaned_ids = self.remove_special_tokens(generated_ids)

        if not cleaned_ids:
            return ""

        return self.processor.decode(cleaned_ids)