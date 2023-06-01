"""Based on MonoTagHMMTransducer model from Wu and Cotterell 2019 (0th-order hard attention)"""

from typing import Optional, Tuple

import torch
from torch import nn

from .. import batches
from . import lstm


class ExactHardMonoEncoderDecoder(lstm.LSTMEncoderDecoder):
    """LSTM encoder-decoder with with exact hard monotonic attention."""

    def __init__(self):
        self.feature_embed = nn.Embedding(
            self.num_features, self.feature_embedding_dim, padding_idx=PAD_IDX
        )
        self.feature_encoder = nn.Linear(
            self.num_features * self.feature_embedding_dim,
            self.feature_embedding_dim
        )

    def decode_step(#self, encoder_out, encoder_mask, symbol, hidden):
        self,
        symbol: torch.Tensor,
        last_hiddens: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #FIXME: I think in wu et all they use the enc hidden states,and use h_i for transition, and c_i for emmission?
        # transition, emission, hidden = super().decode_step(encoder_out, encoder_mask, symbol, last_hiddens)
        src_seq_len, bat_siz = encoder_mask.shape
        embedded = self.target_embeddings(symbol)
        embedded = self.dropout_layer(embedded)
        # FIXME: we need to embed the tags, and then run them through a linear layer and
        #       relu. layer is (num_tags_in_vocab * tag_emb_size) --> tag_emb_size
        tag_embeddings = self.features_encoder(feature_embeddings)
        h_t, last_hiddens = self.decoder(
            torch.cat(embedded, tag_embeddings), last_hiddens
        )

        # Concatenate previous decoder hidden_state to all encoder hidden states
        # This represents all alignments for computing the emission probs.
        # FIXME: encoder_out[0] maybe should be h_i from encoder
        ctx_curr = torch.cat(
            (h_t.unsqueeze(1).expand(-1, src_seq_len, -1), encoder_out[0].transpose(0, 1)),
            dim=2,
        )

        # FIXME: encoder_out[1] maybe should be c_i from encoder
        hs_ = encoder_out[1].transpose(0, 1)
        h_t = h_t.unsqueeze(2)
        score = torch.bmm(hs_, h_t).squeeze(2)
        trans = F.softmax(score, dim=-1) * encoder_mask.transpose(0, 1) + EPSILON
        trans = trans / trans.sum(-1, keepdim=True)
        trans = trans.unsqueeze(1).log()
        trans = trans.expand(bat_siz, src_seq_len, src_seq_len)

        ctx = torch.tanh(self.linear_out(ctx_curr))
        # emission: batch x seq_len x nb_vocab
        emission = F.log_softmax(self.final_out(ctx), dim=-1)

        transition_mask = torch.ones_like(trans[0]).triu().unsqueeze(0)
        transition_mask = (transition_mask - 1) * -np.log(EPSILON)
        transition = transition + transition_mask
        transition = transition - transition.logsumexp(-1, keepdim=True)
        return transition, emission, last_hiddens

    def decode(
        self,
        batch_size: int,
        encoder_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # trg_embed = self.dropout(self.trg_embed(target))
        decoder_hiddens = self.init_hiddens(batch_size, self.decoder_layers)
        # Feed in the first decoder input, as a start tag.
        # -> B x 1.
        decoder_input = (
            torch.tensor(
                [self.start_idx], device=self.device, dtype=torch.long
            )
            .repeat(batch_size)
            .unsqueeze(1)
        )

        initial, transition, emission = None, list(), list()
        num_steps = (
            target.size(1) if target is not None else self.max_target_length
        )
        for t in range(num_steps):
            trans, emiss, decoder_hiddens = self.decode_step(
                decoder_input, decoder_hiddens, encoder_out, encoder_mask
            )

            predictions.append(output.squeeze(1))
            # If we have a target (training) then the next input is the gold
            # symbol for this step (i.e., teacher forcing).
            if target is not None:
                # TODO: Here I think we want to sum over the each token value, which is a sum of
                #       log(transition prob) + log(emiission prob)
                decoder_input = target[:, t].unsqueeze(1)
            # Otherwise, it must be inference time and we pass the top pred to
            # the next next timestep (i.e., student forcing, greedy decoding).
            else:
                output = output + trans
                output = output.logsumexp(dim=-1, keepdim=True)
                decoder_input = self._get_predicted(output)
                log_wordprob = output + emiss.transpose(1, 2)
                log_wordprob = log_wordprob.logsumexp(dim=-1)
                word = torch.max(log_wordprob, dim=-1)[1]
                # Updates to track which sequences have decoded an EOS.
                finished = torch.logical_or(
                    finished, (decoder_input == self.end_idx)
                )
                # Breaks when all batches predicted an EOS symbol.
                if finished.all():
                    break

            if t == 0:
                initial = trans[:, 0].unsqueeze(1)
            else:
                transition += [trans] 
        emission += [emiss] 
        transition = torch.stack(transition)
        emission = torch.stack(emission)
        return initial, transition, emission