from __future__ import annotations
import math
import torch
from torch import Tensor, nn

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class PositionalEncoding(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_length: int = 64,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if d_model % 2 != 0:
            raise ValueError("d_model must be even.")

        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                start=0,
                end=d_model,
                step=2,
                dtype=torch.float32,
            )
            * (-math.log(10_000.0) / d_model)
        )

        encoding = torch.zeros(
            max_length,
            d_model,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        
        if embeddings.ndim != 3:
            raise ValueError(
                "Expected embeddings with shape "
                "(batch_size, sequence_length, d_model)."
            )

        if embeddings.size(-1) != self.encoding.size(-1):
            raise ValueError(
                "Embedding dimension does not match positional "
                f"encoding dimension: {embeddings.size(-1)} != "
                f"{self.encoding.size(-1)}"
            )

        sequence_length = embeddings.size(1)

        if sequence_length > self.encoding.size(1):
            raise ValueError(
                "Input sequence is longer than the positional "
                f"encoding limit: {sequence_length} > "
                f"{self.encoding.size(1)}"
            )

        return self.dropout(
            embeddings + self.encoding[:, :sequence_length]
        )


class TransformerNMT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_sequence_length: int = 64,
        pad_id: int = PAD_ID,
        unk_id: int = UNK_ID,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
    ) -> None:
        super().__init__()

        if vocab_size <= 4:
            raise ValueError(
                "vocab_size must be greater than the special-token count."
            )

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads. "
                f"Received d_model={d_model}, "
                f"num_heads={num_heads}."
            )

        if num_encoder_layers <= 0:
            raise ValueError(
                "num_encoder_layers must be positive."
            )

        if num_decoder_layers <= 0:
            raise ValueError(
                "num_decoder_layers must be positive."
            )

        if feedforward_dim <= 0:
            raise ValueError(
                "feedforward_dim must be positive."
            )

        if max_sequence_length < 3:
            raise ValueError(
                "max_sequence_length must be at least 3."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)."
            )

        special_ids = {
            "pad_id": pad_id,
            "unk_id": unk_id,
            "bos_id": bos_id,
            "eos_id": eos_id,
        }

        for name, token_id in special_ids.items():
            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"{name} is outside the vocabulary: "
                    f"{token_id}"
                )

        if len(set(special_ids.values())) != 4:
            raise ValueError(
                "pad_id, unk_id, bos_id, and eos_id must be distinct."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.max_sequence_length = max_sequence_length

        self.source_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.target_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.source_position = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_length=max_sequence_length,
        )

        self.target_position = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_length=max_sequence_length,
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
        )

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Apply Xavier uniform initialization to matrix parameters.

        PAD embeddings are then explicitly reset to zero.
        """
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        with torch.no_grad():
            self.source_embedding.weight[self.pad_id].zero_()
            self.target_embedding.weight[self.pad_id].zero_()

    @staticmethod
    def causal_mask(
        length: int,
        device: torch.device,
    ) -> Tensor:
        """
        Create a Boolean causal mask for decoder self-attention.

        False means attention is allowed.
        True means attention is blocked.

        Future target positions are blocked.
        """
        if length <= 0:
            raise ValueError(
                "Causal-mask length must be positive."
            )

        return torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def _validate_token_ids(
        self,
        token_ids: Tensor,
        name: str,
    ) -> None:
        """Validate token-ID tensor shape, type, and value range."""
        if token_ids.ndim != 2:
            raise ValueError(
                f"{name} must have shape "
                "(batch_size, sequence_length)."
            )

        if token_ids.dtype != torch.long:
            raise TypeError(
                f"{name} must have dtype torch.long; "
                f"received {token_ids.dtype}."
            )

        if token_ids.numel() == 0:
            raise ValueError(f"{name} cannot be empty.")

        if token_ids.size(1) > self.max_sequence_length:
            raise ValueError(
                f"{name} exceeds maximum sequence length: "
                f"{token_ids.size(1)} > "
                f"{self.max_sequence_length}"
            )

        if token_ids.min().item() < 0:
            raise ValueError(
                f"{name} contains a negative token ID."
            )

        if token_ids.max().item() >= self.vocab_size:
            raise ValueError(
                f"{name} contains an ID outside the vocabulary. "
                f"Maximum allowed ID is {self.vocab_size - 1}."
            )

    @staticmethod
    def _validate_padding_mask(
        mask: Tensor | None,
        token_ids: Tensor,
        name: str,
    ) -> None:
        """Validate an optional Boolean key-padding mask."""
        if mask is None:
            return

        if mask.dtype != torch.bool:
            raise TypeError(
                f"{name} must have dtype torch.bool; "
                f"received {mask.dtype}."
            )

        if mask.shape != token_ids.shape:
            raise ValueError(
                f"{name} shape must match token IDs. "
                f"Expected {tuple(token_ids.shape)}, "
                f"received {tuple(mask.shape)}."
            )

        if mask.device != token_ids.device:
            raise ValueError(
                f"{name} and token IDs must be on the same device."
            )

    def encode(
        self,
        source: Tensor,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        
        self._validate_token_ids(source, "source")

        self._validate_padding_mask(
            source_padding_mask,
            source,
            "source_padding_mask",
        )

        source_embeddings = self.source_embedding(source)
        source_embeddings = (
            source_embeddings * math.sqrt(self.d_model)
        )

        source_embeddings = self.source_position(
            source_embeddings
        )

        return self.transformer.encoder(
            src=source_embeddings,
            src_key_padding_mask=source_padding_mask,
        )

    def decode(
        self,
        target_input: Tensor,
        memory: Tensor,
        target_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
    
        self._validate_token_ids(
            target_input,
            "target_input",
        )

        self._validate_padding_mask(
            target_padding_mask,
            target_input,
            "target_padding_mask",
        )

        if memory.ndim != 3:
            raise ValueError(
                "memory must have shape "
                "(batch_size, source_length, d_model)."
            )

        if memory.size(0) != target_input.size(0):
            raise ValueError(
                "Memory and target batch sizes do not match."
            )

        if memory.size(-1) != self.d_model:
            raise ValueError(
                "Memory embedding dimension does not match d_model."
            )

        if memory.device != target_input.device:
            raise ValueError(
                "memory and target_input must be on the same device."
            )

        if memory_padding_mask is not None:
            if memory_padding_mask.dtype != torch.bool:
                raise TypeError(
                    "memory_padding_mask must have dtype torch.bool."
                )

            expected_shape = (
                memory.size(0),
                memory.size(1),
            )

            if memory_padding_mask.shape != expected_shape:
                raise ValueError(
                    "memory_padding_mask shape does not match memory. "
                    f"Expected {expected_shape}, received "
                    f"{tuple(memory_padding_mask.shape)}."
                )

            if memory_padding_mask.device != memory.device:
                raise ValueError(
                    "memory_padding_mask and memory must be on "
                    "the same device."
                )

        target_embeddings = self.target_embedding(
            target_input
        )

        target_embeddings = (
            target_embeddings * math.sqrt(self.d_model)
        )

        target_embeddings = self.target_position(
            target_embeddings
        )

        target_mask = self.causal_mask(
            length=target_input.size(1),
            device=target_input.device,
        )

        decoded = self.transformer.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        return self.output_projection(decoded)

    def forward(
        self,
        source: Tensor,
        target_input: Tensor,
        source_padding_mask: Tensor | None = None,
        target_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Run a teacher-forcing training forward pass.

        Args:
            source: Source token IDs.
            target_input: Target IDs excluding the final label token.
            source_padding_mask: True at source PAD locations.
            target_padding_mask: True at target PAD locations.

        Returns:
            Vocabulary logits for every target position.
        """
        memory = self.encode(
            source=source,
            source_padding_mask=source_padding_mask,
        )

        return self.decode(
            target_input=target_input,
            memory=memory,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=source_padding_mask,
        )

    @torch.inference_mode()
    def greedy_decode(
        self,
        source: Tensor,
        source_padding_mask: Tensor | None = None,
        max_length: int | None = None,
    ) -> Tensor:
        """
        Translate source IDs using greedy autoregressive decoding.

        Generation begins with BOS. It stops when every sequence has
        produced EOS or reaches max_length.

        Returns:
            Generated target IDs including BOS and, where generated, EOS.
        """
        self._validate_token_ids(source, "source")

        self._validate_padding_mask(
            source_padding_mask,
            source,
            "source_padding_mask",
        )

        if max_length is None:
            max_length = self.max_sequence_length

        if max_length < 2:
            raise ValueError(
                "max_length must be at least 2."
            )

        if max_length > self.max_sequence_length:
            raise ValueError(
                "max_length cannot exceed the positional encoding "
                f"limit ({self.max_sequence_length})."
            )

        was_training = self.training
        self.eval()

        try:
            memory = self.encode(
                source=source,
                source_padding_mask=source_padding_mask,
            )

            batch_size = source.size(0)

            generated = torch.full(
                size=(batch_size, 1),
                fill_value=self.bos_id,
                dtype=torch.long,
                device=source.device,
            )

            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=source.device,
            )

            for _ in range(max_length - 1):
                logits = self.decode(
                    target_input=generated,
                    memory=memory,
                    target_padding_mask=generated.eq(self.pad_id),
                    memory_padding_mask=source_padding_mask,
                )

                next_token_scores = logits[:, -1, :].clone()

                # Do not generate padding, unknown, or repeated BOS.
                next_token_scores[:, self.pad_id] = float("-inf")
                next_token_scores[:, self.unk_id] = float("-inf")
                next_token_scores[:, self.bos_id] = float("-inf")

                next_token = next_token_scores.argmax(dim=-1)

                next_token = torch.where(
                    finished,
                    torch.full_like(
                        next_token,
                        self.pad_id,
                    ),
                    next_token,
                )

                generated = torch.cat(
                    [generated, next_token.unsqueeze(1)],
                    dim=1,
                )

                finished |= next_token.eq(self.eos_id)

                if bool(finished.all()):
                    break

            return generated

        finally:
            if was_training:
                self.train()

if __name__ == "__main__":
    from pathlib import Path

    from dataset import create_dataloaders

    print("Testing Transformer forward pass...")

    project_root = Path(__file__).resolve().parent.parent

    tokenized_dir = (
        project_root
        / "tokenizer"
        / "tokenized"
    )

    train_loader, _, _ = create_dataloaders(
        tokenized_dir=tokenized_dir,
        batch_size=4,
        max_source_length=64,
        max_target_length=64,
    )

    batch = next(iter(train_loader))

    model = TransformerNMT(
        vocab_size=10_000,
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        feedforward_dim=1024,
        dropout=0.1,
        max_sequence_length=64,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    target_input = batch.target[:, :-1]

    logits = model(
        source=batch.source,
        target_input=target_input,
        source_padding_mask=batch.source_padding_mask,
        target_padding_mask=target_input.eq(PAD_ID),
    )

    expected_shape = (
        batch.source.size(0),
        target_input.size(1),
        10_000,
    )

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected logits shape. "
            f"Expected {expected_shape}, received "
            f"{tuple(logits.shape)}."
        )

    generated = model.greedy_decode(
        source=batch.source,
        source_padding_mask=batch.source_padding_mask,
        max_length=16,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Source shape:       {tuple(batch.source.shape)}")
    print(f"Target shape:       {tuple(batch.target.shape)}")
    print(f"Decoder input:      {tuple(target_input.shape)}")
    print(f"Output logits:      {tuple(logits.shape)}")
    print(f"Generated IDs:      {tuple(generated.shape)}")
    print(f"Model parameters:   {parameter_count:,}")
    print("✓ Transformer forward-pass sanity check passed.")