import math
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from bert.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class BertDataSetHead512(Dataset):
    def __init__(self, corpus_path):
        self.labels = []
        self.corpus_path = corpus_path
        self.descriptions = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.labels.append(int(line[0]))
                        self.descriptions.append(line[1].split(' ')[:511])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item]
        tokens_id = [7549] + [int(x) for x in token_text]
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(7550)
        segment_ids = [1 if x!=7550 else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance


class BertTestSetHead512(Dataset):
    def __init__(self, test_path):
        self.corpus_path = test_path
        self.labels = []
        self.descriptions = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    line = line.split(',')
                    if line[0] and line[1]:
                        self.labels.append(int(line[0]))
                        self.descriptions.append(line[1].split(' ')[:511])

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        output = {}
        label_text = self.labels[item]
        token_text = self.descriptions[item]
        tokens_id = [7549] + [int(x) for x in token_text]
        if len(tokens_id) < SentenceLength:
            for i in range(SentenceLength - len(tokens_id)):
                tokens_id.append(7550)
        segment_ids = [1 if x!=7550 else 0 for x in tokens_id]
        output['input_token_ids'] = tokens_id
        output['segment_ids'] = segment_ids
        output['token_ids_labels'] = label_text
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance
