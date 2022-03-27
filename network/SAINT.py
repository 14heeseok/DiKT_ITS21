import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import PAD_INDEX, RPAD_INDEX
from config import ARGS
from network.util_network import get_pad_mask, get_subsequent_mask, clones
from network.SAKT import *


class TransformerBlock(nn.Module):
    """
    Single Transformer block of SAINT
    """
    def __init__(self, hidden_dim, num_head, dropout):
        super().__init__()
        self._self_attn = MultiHeadedAttention(num_head, hidden_dim, dropout)
        self._ffn = PositionwiseFeedForward(hidden_dim, hidden_dim, dropout)
        self._layernorms = clones(nn.LayerNorm(hidden_dim, eps=1e-8), 2)

    def forward(self, query, key, value, mask):
        output = self._self_attn(query=query, key=key, value=value, mask=mask)
        output = self._layernorms[0](key + output)
        output = self._layernorms[1](output + self._ffn(output))
        return output


class SAINT(nn.Module):
    def __init__(self, hidden_dim, question_num,
                 num_enc_layers, num_dec_layers, num_head, dropout):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._question_num = question_num

        # Encoder blocks
        self._encoder = clones(TransformerBlock(hidden_dim, num_head, dropout),
                               num_enc_layers)
        # Decoder blocks
        self._decoder = clones(TransformerBlock(hidden_dim, num_head, dropout),
                               num_dec_layers)

        # Embedding layers
        self._positional_embedding = nn.Embedding(
            ARGS.seq_size+2, hidden_dim, padding_idx=PAD_INDEX)
        self._question_embedding = nn.Embedding(
            question_num+1, hidden_dim, padding_idx=PAD_INDEX)
        self._response_embedding = nn.Embedding(
            2+2, hidden_dim, padding_idx=RPAD_INDEX)
        self._prediction = nn.Linear(hidden_dim, 1)

    def _transform_interaction_to_question_id(self, interaction):
        """
        get question_id from interaction index
        if interaction index is a number in [0, question_num], then leave it as-is
        if interaction index is bigger than question_num (in [question_num + 1, 2 * question_num]
        then subtract question_num
        interaction: integer tensor of shape (batch_size, sequence_size)
        """
        return interaction - self._question_num *\
             (interaction > self._question_num).long()

    def _transform_interaction_to_response_id(self, interaction):
        question_id = interaction - self._question_num *\
            (interaction > self._question_num).long()
        pads = interaction == 0
        response_id = (interaction > self._question_num).long()
        response_id = response_id.masked_fill(pads, RPAD_INDEX)
        return response_id.long()

    def _get_position_index(self, input, for_decoder=False):
        batch_size = input.shape[0]
        seq_len = ARGS.seq_size
        position_indices = []
        if not for_decoder:
            for i in range(batch_size):
                question_id = input
                non_padding_num = (question_id[i] != PAD_INDEX).sum(-1).item()
                position_index = [0] * (seq_len - non_padding_num) +\
                    list(range(1, non_padding_num+1))
                position_indices.append(position_index)
        else:
            for i in range(batch_size):
                response_id = input
                non_padding_num = (response_id[i] != RPAD_INDEX).sum(-1).item()
                position_index = [0] * min(seq_len - non_padding_num, seq_len-1) +\
                    [seq_len+1] + list(range(1, non_padding_num))
                position_indices.append(position_index)
        return torch.tensor(position_indices, dtype=int).to(ARGS.device)

    def forward(self, interaction_id, target_id):
        question_id =\
            self._transform_interaction_to_question_id(interaction_id)
        question_id = torch.cat([question_id[:, 1:], target_id], dim=-1)
        question_vector = self._question_embedding(question_id)

        position_index = self._get_position_index(question_id)
        position_vector = self._positional_embedding(position_index)

        response_id =\
            self._transform_interaction_to_response_id(interaction_id)
        response_vector = self._response_embedding(response_id)

        response_position_index =\
            self._get_position_index(response_id, for_decoder=True)
        response_position_vector =\
            self._positional_embedding(response_position_index)

        x = question_vector + position_vector
        decoder_input = response_vector + response_position_vector

        mask = get_pad_mask(question_id, PAD_INDEX) &\
            get_subsequent_mask(question_id)

        # encoder forward pass
        for layer in self._encoder:
            x = layer(query=x, key=x, value=x, mask=mask)
        encoder_out = x
        # decoder forward pass
        for n, layer in enumerate(self._decoder):
            if n == 0:
                decoder_out = layer(query=decoder_input, key=decoder_input,
                                    value=decoder_input, mask=mask)
            else:
                decoder_out = layer(query=decoder_out, key=encoder_out,
                                    value=encoder_out, mask=mask)

        output = self._prediction(decoder_out)[:, -1, :]
        return output
