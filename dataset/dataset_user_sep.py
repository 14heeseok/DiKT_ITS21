import torch
from torch.utils.data import Dataset

from config import ARGS
from constant import *


class UserSepDataset(Dataset):

    def __init__(self, name, sample_infos, dataset_name='ASSISTments2009'):
        self._name = name  # train, val, test
        self._sample_infos = sample_infos  # list of (user_path, target_index)
        self._dataset_name = dataset_name

    def get_sequence(self, sample):
        user_path, target_index = sample
        with open(user_path, 'r') as f:
            data = f.readlines()[1:]  # header exists
            data = data[:target_index+1]
            user_data_length = len(data)

        if user_data_length > ARGS.seq_size + 1:
            data = data[-(ARGS.seq_size + 1):]
            pad_counts = 0
        else:
            pad_counts = ARGS.seq_size + 1 - user_data_length

        input_list = []
        for idx, line in enumerate(data):
            line = line.rstrip().split(',')
            tag_id = int(line[0])
            is_correct = int(line[1])

            if idx == len(data) - 1:
                last_is_correct = is_correct
                target_id = tag_id
            else:
                if is_correct:
                    input_list.append(tag_id)
                else:
                    input_list.append(tag_id + QUESTION_NUM[self._dataset_name])

        paddings = [PAD_INDEX] * pad_counts
        input_list = input_list + paddings
        assert len(input_list) == ARGS.seq_size, "sequence size error"

        if ARGS.model == 'DiKT':
            right_list = [tag_id if tag_id < QUESTION_NUM[self._dataset_name] else PAD_INDEX
                          for tag_id in input_list]
            wrong_list = [tag_id if tag_id > QUESTION_NUM[self._dataset_name] else PAD_INDEX
                          for tag_id in input_list]
            return_input = {'right': torch.Tensor(right_list).long(),
                            'wrong': torch.Tensor(wrong_list).long()}
        else:
            return_input = torch.Tensor(input_list).long()

        return {
            'label': torch.Tensor([last_is_correct]).long(),
            'input': return_input,
            'target_id': torch.Tensor([target_id]).long()
        }

    def __repr__(self):
        return f'{self._name}: # of samples: {len(self._sample_infos)}'

    def __len__(self):
        return len(self._sample_infos)

    def __getitem__(self, index):
        return self.get_sequence(self._sample_infos[index])
