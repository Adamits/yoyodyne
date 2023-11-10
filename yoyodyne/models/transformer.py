"""Transformer model classes."""

import argparse
from typing import Optional

import torch
from torch import nn

from .. import data, defaults
from . import base, modules


class TransformerEncoderDecoder(base.BaseEncoderDecoder):
    """Transformer encoder-decoder."""

    # Model arguments.
    source_attention_heads: int
    # Constructed inside __init__.
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        source_attention_heads=defaults.SOURCE_ATTENTION_HEADS,
        **kwargs,
    ):
        """Initializes the encoder-decoder with attention.

        Args:
            source_attention_heads (int).
            max_source_length (int).
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        self.source_attention_heads = source_attention_heads
        super().__init__(
            *args, source_attention_heads=source_attention_heads, **kwargs
        )
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )

    def get_decoder(self):
        return modules.transformer.TransformerDecoder(
            pad_idx=self.pad_idx,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            num_embeddings=self.target_vocab_size,
            decoder_input_size=self.source_encoder.output_size,
            dropout=self.dropout,
            embedding_size=self.embedding_size,
            source_attention_heads=self.source_attention_heads,
            max_source_length=self.max_source_length,
            layers=self.decoder_layers,
            hidden_size=self.hidden_size,
        )

    def _decode_greedy(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes the output sequence greedily.

        Args:
            encoder_hidden (torch.Tensor): Hidden states from the encoder.
            source_mask (torch.Tensor): Mask for the encoded source tokens.
            targets (torch.Tensor, optional): The optional target tokens,
                which is only used for early stopping during validation
                if the decoder has predicted [EOS] for every sequence in
                the batch.

        Returns:
            torch.Tensor: predictions from the decoder.
        """
        # The output distributions to be returned.
        outputs = []
        batch_size = encoder_hidden.size(0)
        # The predicted symbols at each iteration.
        predictions = [
            torch.tensor(
                [self.start_idx for _ in range(encoder_hidden.size(0))],
                device=self.device,
            )
        ]
        # Tracking when each sequence has decoded an EOS.
        finished = torch.zeros(batch_size, device=self.device)
        for _ in range(self.max_target_length):
            target_tensor = torch.stack(predictions, dim=1)
            # Uses a dummy mask of all ones.
            target_mask = torch.ones_like(target_tensor, dtype=torch.float)
            target_mask = target_mask == 0
            decoder_output = self.decoder(
                encoder_hidden, source_mask, target_tensor, target_mask
            ).output
            logits = self.classifier(decoder_output)
            last_output = logits[:, -1, :]  # Ignores EOS.
            outputs.append(last_output)
            # -> B x 1 x 1
            _, pred = torch.max(last_output, dim=1)
            predictions.append(pred)
            # Updates to track which sequences have decoded an EOS.
            finished = torch.logical_or(
                finished, (predictions[-1] == self.end_idx)
            )
            # Breaks when all sequences have predicted an EOS symbol. If we
            # have a target (and are thus computing loss), we only break when
            # we have decoded at least the the same number of steps as the
            # target length.
            if finished.all():
                if targets is None or len(outputs) >= targets.size(-1):
                    break
        # -> B x seq_len x target_vocab_size.
        return torch.stack(outputs).transpose(0, 1)

    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.
        """
        if self.training and self.teacher_forcing:
            assert (
                batch.target.padded is not None
            ), "Teacher forcing requested but no target provided"
            # Initializes the start symbol for decoding.
            starts = (
                torch.tensor(
                    [self.start_idx], device=self.device, dtype=torch.long
                )
                .repeat(batch.target.padded.size(0))
                .unsqueeze(1)
            )
            target_padded = torch.cat((starts, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (starts == self.pad_idx, batch.target.mask), dim=1
            )
            encoder_output = self.source_encoder(batch.source).output
            decoder_output = self.decoder(
                encoder_output, batch.source.mask, target_padded, target_mask
            ).output
            logits = self.classifier(decoder_output)
            output = logits[:, :-1, :]  # Ignore EOS.
        else:
            encoder_output = self.source_encoder(batch.source).output
            # -> B x seq_len x output_size.
            output = self._decode_greedy(
                encoder_output,
                batch.source.mask,
                batch.target.padded if batch.target else None,
            )
        return output

    @property
    def name(self) -> str:
        return "transformer"

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        """Adds transformer configuration options to the argument parser.

        These are only needed at training time.

        Args:
            parser (argparse.ArgumentParser).
        """
        parser.add_argument(
            "--source_attention_heads",
            type=int,
            default=defaults.SOURCE_ATTENTION_HEADS,
            help="Number of attention heads "
            "(transformer-backed architectures only. Default: %(default)s.",
        )
        parser.add_argument(
            "--features_attention_heads",
            type=int,
            default=defaults.FEATURES_ATTENTION_HEADS,
            help="Number of features attention heads "
            "(transformer-backed pointer-generator only. "
            "Default: %(default)s.",
        )


class DecoderOnlyTransformer(base.BaseDecoderOnly):
    """Transformer decoder only model."""

    # Constructed inside __init__.
    classifier: nn.Linear

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the encoder-decoder with attention.

        Args:
            max_source_length (int).
            *args: passed to superclass.
            **kwargs: passed to superclass.
        """
        super().__init__(
            *args, **kwargs
        )
        self.classifier = nn.Linear(
            self.embedding_size, self.target_vocab_size
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.decoder_input_size,
            dim_feedforward=self.hidden_size,
            nhead=self.source_attention_heads,
            dropout=self.dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.embedding_size),
        )

    def _get_prefix_mask():
        # Maybe use https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/52/files as reference
        # but they basically do what youd expect.
        pass

    def decode(
        self,
        encoder_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        target_embedding = self.embed(target)
        target_sequence_length = target_embedding.size(1)
        # TODO: instead we want a causal-prefix mask.
        # Something like causal_mask[:, :prefix_length]
        # ISSUE: we will have variable prefix lengths per batch right?.
        #       I guess we may want to handle this in dataloading.
        # -> seq_len x seq_len.
        causal_mask = self.generate_square_subsequent_mask(
            target_sequence_length
        ).to(self.device)

        # -> B x seq_len x d_model.
        output = self.decoder(
            target_embedding,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_mask,
            tgt_key_padding_mask=target_mask,
        )
        return base.ModuleOutput(output, embeddings=target_embedding)
    
    def forward(
        self,
        batch: data.PaddedBatch,
    ) -> torch.Tensor:
        """Runs the encoder-decoder.

        Args:
            batch (data.PaddedBatch).

        Returns:
            torch.Tensor.
        """
        if self.training and self.teacher_forcing:
            assert (
                batch.target.padded is not None
            ), "Teacher forcing requested but no target provided"
            # Initializes the start symbol for decoding.
            starts = (
                torch.tensor(
                    [self.start_idx], device=self.device, dtype=torch.long
                )
                .repeat(batch.target.padded.size(0))
                .unsqueeze(1)
            )
            target_padded = torch.cat((starts, batch.target.padded), dim=1)
            target_mask = torch.cat(
                (starts == self.pad_idx, batch.target.mask), dim=1
            )
            decoder_output = self.decoder(
                encoder_output, batch.source.mask, target_padded, target_mask
            ).output
            logits = self.classifier(decoder_output)
            output = logits[:, :-1, :]  # Ignore EOS.
        else:
            encoder_output = self.source_encoder(batch.source).output
            # -> B x seq_len x output_size.
            output = self._decode_greedy(
                encoder_output,
                batch.source.mask,
                batch.target.padded if batch.target else None,
            )
        return output

    @property
    def name(self) -> str:
        return "decoder-only transformer"