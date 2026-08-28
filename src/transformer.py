from __future__ import annotations

import math

import torch
from torch import Tensor, nn


PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_length: int = 512,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even.")

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
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
        sequence_length = embeddings.size(1)

        if sequence_length > self.encoding.size(1):
            raise ValueError(
                "Input sequence is longer than positional encoding limit: "
                f"{sequence_length} > {self.encoding.size(1)}"
            )

        embeddings = embeddings + self.encoding[:, :sequence_length]
        return self.dropout(embeddings)


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
        max_sequence_length: int = 256,
        pad_id: int = PAD_ID,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_sequence_length = max_sequence_length

        self.source_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        self.target_embedding = nn.Embedding(
            vocab_size,
            d_model,
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
            d_model,
            vocab_size,
        )

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def causal_mask(
        length: int,
        device: torch.device,
    ) -> Tensor:
        return nn.Transformer.generate_square_subsequent_mask(
            length,
            device=device,
        )

    def encode(
        self,
        source: Tensor,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        source_embeddings = self.source_embedding(source)
        source_embeddings = source_embeddings * math.sqrt(self.d_model)
        source_embeddings = self.source_position(source_embeddings)

        return self.transformer.encoder(
            source_embeddings,
            src_key_padding_mask=source_padding_mask,
        )

    def decode(
        self,
        target: Tensor,
        memory: Tensor,
        target_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        target_embeddings = self.target_embedding(target)
        target_embeddings = target_embeddings * math.sqrt(self.d_model)
        target_embeddings = self.target_position(target_embeddings)

        target_mask = self.causal_mask(
            target.size(1),
            target.device,
        )

        decoded = self.transformer.decoder(
            target_embeddings,
            memory,
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
        memory = self.encode(
            source,
            source_padding_mask=source_padding_mask,
        )

        return self.decode(
            target_input,
            memory,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=source_padding_mask,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        source: Tensor,
        source_padding_mask: Tensor | None = None,
        max_length: int = 256,
    ) -> Tensor:
        self.eval()

        memory = self.encode(
            source,
            source_padding_mask=source_padding_mask,
        )

        batch_size = source.size(0)
        generated = torch.full(
            (batch_size, 1),
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
                generated,
                memory,
                target_padding_mask=generated.eq(self.pad_id),
                memory_padding_mask=source_padding_mask,
            )

            next_token = logits[:, -1, :].argmax(dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.pad_id),
                next_token,
            )

            generated = torch.cat(
                [generated, next_token.unsqueeze(1)],
                dim=1,
            )

            finished = finished | next_token.eq(self.eos_id)

            if bool(finished.all()):
                break

        return generated