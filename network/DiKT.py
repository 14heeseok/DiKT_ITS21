from math import sqrt

import torch
import torch.nn as nn

from constant import PAD_INDEX
from config import ARGS


class DiKT(nn.Module):
    """
    B version of deep pfa model
    - sigmoid applied to sucess and failed attempts
    - random key memory replaced with fully connected layer
    """
    def __init__(self, key_dim, value_dim, summary_dim, question_num, concept_num):
        super().__init__()
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._summary_dim = summary_dim
        self._question_num = question_num
        self._concept_num = concept_num

        # embedding layers
        self._question_embedding = nn.Embedding(
            num_embeddings=question_num+1, embedding_dim=key_dim,
            padding_idx=PAD_INDEX)
        self._interaction_embedding = nn.Embedding(
            num_embeddings=2*question_num+1, embedding_dim=value_dim,
            padding_idx=PAD_INDEX)

        # FC layers
        self._key_layer = nn.Linear(
            in_features=key_dim, out_features=concept_num, bias=False)
        self._erase_layer = nn.Linear(
            in_features=value_dim, out_features=value_dim)
        self._add_layer = nn.Linear(
            in_features=value_dim, out_features=value_dim)
        self._right_summary_layer = nn.Linear(
            in_features=value_dim+key_dim, out_features=summary_dim)
        self._wrong_summary_layer = nn.Linear(
            in_features=value_dim+key_dim, out_features=summary_dim)

        # activation functions
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()
        self._softmax = nn.Softmax(dim=-1)

        # PFA layers
        self._success_layer = nn.Linear(
            in_features=summary_dim, out_features=1)
        self._failure_layer = nn.Linear(
            in_features=summary_dim, out_features=1)
        self._difficulty_layer = nn.Linear(
            in_features=key_dim, out_features=1)

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

    def _init_value_memory(self):
        """
        initialize value memory matrix
        follows initialization that used in the following NMT implementation:
        https://github.com/loudinthecloud/pytorch-ntm/blob/master/ntm/memory.py
        """
        # value memory matrix, transposed
        self._right_value_memory = torch.Tensor(
            self._value_dim, self._concept_num).to(ARGS.device)
        self._wrong_value_memory = torch.Tensor(
            self._value_dim, self._concept_num).to(ARGS.device)

        stdev = 1 / (sqrt(self._concept_num + self._key_dim))
        nn.init.uniform_(self._right_value_memory, -stdev, stdev)
        nn.init.uniform_(self._wrong_value_memory, -stdev, stdev)

        # (batch_size, key_dim, concept_num)
        self._right_value_memory = self._right_value_memory.clone().repeat(
            self._batch_size, 1, 1)
        self._wrong_value_memory = self._wrong_value_memory.clone().repeat(
            self._batch_size, 1, 1)

    def _compute_correlation_weight(self, question_id):
        """
        compute correlation weight of a given question with key memory matrix
        question_id: integer tensor of shape (batch_size)
        """
        question_vector = self._question_embedding(question_id).to(ARGS.device)
        return self._softmax(self._key_layer(question_vector)).to(ARGS.device)

    def _read(self, question_id):
        """
        read process - get read content vector from question_id and value memory matrix
        question_id: (batch_size)
        """
        question_id = question_id.squeeze(-1)
        correlation_weight = self._compute_correlation_weight(question_id)
        right_read_content =\
            torch.matmul(self._right_value_memory,
                         correlation_weight.unsqueeze(-1)).squeeze(-1)
        wrong_read_content =\
            torch.matmul(self._wrong_value_memory,
                         correlation_weight.unsqueeze(-1)).squeeze(-1)
        return right_read_content.to(ARGS.device),\
            wrong_read_content.to(ARGS.device)

    def _write(self, right_interaction, wrong_interaction):
        """
        write process - update value memory matrix
        interaction: (batch_size)
        """
        right_interaction_vector =\
            self._interaction_embedding(right_interaction)
        wrong_interaction_vector =\
            self._interaction_embedding(wrong_interaction)
        right_question_id =\
            self._transform_interaction_to_question_id(right_interaction)
        wrong_question_id =\
            self._transform_interaction_to_question_id(wrong_interaction)

        self._prev_right_value_memory = self._right_value_memory
        self._prev_wrong_value_memory = self._wrong_value_memory

        e_right = self._sigmoid(self._erase_layer(right_interaction_vector))
        a_right = self._tanh(self._add_layer(right_interaction_vector))
        e_wrong = self._sigmoid(self._erase_layer(wrong_interaction_vector))
        a_wrong = self._tanh(self._add_layer(wrong_interaction_vector))

        w_right = self._compute_correlation_weight(right_question_id)
        erase_right = torch.matmul(w_right.unsqueeze(-1), e_right.unsqueeze(1))
        erase_right = torch.transpose(erase_right, 1, 2)
        add_right = torch.matmul(w_right.unsqueeze(-1), a_right.unsqueeze(1))
        add_right = torch.transpose(add_right, 1, 2)
        self._right_value_memory =\
            self._prev_right_value_memory * (1 - erase_right) + add_right

        w_wrong = self._compute_correlation_weight(wrong_question_id)
        erase_wrong = torch.matmul(w_wrong.unsqueeze(-1), e_wrong.unsqueeze(1))
        erase_wrong = torch.transpose(erase_wrong, 1, 2)
        add_wrong = torch.matmul(w_wrong.unsqueeze(-1), a_wrong.unsqueeze(1))
        add_wrong = torch.transpose(add_wrong, 1, 2)
        self._wrong_value_memory =\
            self._prev_wrong_value_memory * (1 - erase_wrong) + add_wrong

    def forward(self, input, target_id):
        """
        get output of the model (before taking sigmoid)
        input: integer tensor of shape (batch_size, sequence_size)
        target_id: integer tensor of shape (batch_size)
        """
        # split input data into right and wrong
        wrong_input, right_input = input['wrong'], input['right']

        # initialize value memory matrix
        batch_size = right_input.shape[0]
        self._batch_size = batch_size
        self._init_value_memory()

        # repeat write process seq_size many times with input
        for i in range(ARGS.seq_size):
            right_interaction = right_input[:, i]  # (batch_size)-each qa in batch
            wrong_interaction = wrong_input[:, i]
            self._write(right_interaction, wrong_interaction)

        # read process
        question_vector = self._question_embedding(target_id)
        question_vector = question_vector.squeeze(1)
        right_read_content, wrong_read_content = self._read(target_id)

        right_summary_vector = self._right_summary_layer(
            torch.cat((right_read_content, question_vector), dim=-1))
        right_summary_vector = self._tanh(right_summary_vector)
        wrong_summary_vector = self._wrong_summary_layer(
            torch.cat((wrong_read_content, question_vector), dim=-1))
        wrong_summary_vector = self._tanh(wrong_summary_vector)

        success_level = self._success_layer(right_summary_vector)
        success_level = self._tanh(success_level)

        failure_level = self._failure_layer(wrong_summary_vector)
        failure_level = self._tanh(failure_level)

        difficulty_level = self._difficulty_layer(question_vector)
        difficulty_level = self._tanh(difficulty_level)

        success_count = torch.sum(right_input.clamp(0, 1))
        failure_count = torch.sum(wrong_input.clamp(0, 1))
        output = success_level * self._sigmoid(success_count.float()) +\
            failure_level * self._sigmoid(failure_count.float()) -\
            difficulty_level*2  # scaled to range (-4, 4)
        return output
