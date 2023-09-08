# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


def build_reward_dataset(config):
    def load_dataset(data_frame, tokenizer, pad_size=32):
        contents = []
        for index, row in tqdm(data_frame.iterrows(), total=data_frame.shape[0]):
            prompt_ = row['gpt_prompt']
            content_ = row['content']
            if pd.isna(prompt_):
                continue
            if pd.isna(content_):
                continue

            # parts = prompt_.split('<START>')
            #
            # # 如果 '<START>' 存在于字符串中，那么就获取 '<START>' 之后的部分
            # if len(parts) > 1:
            #     prompt_ = parts[1]
            # else:
            #     print(f"not found <START> with prompt:\n{prompt_}")

            content = prompt_ + ' ' + content_
            label = row['is_accepted']

            token_ids = tokenizer(
                content,
                padding="max_length",
                truncation=True,
                max_length=pad_size
            )
            input_ids = token_ids.input_ids
            mask = token_ids.attention_mask
            seq_len = np.count_nonzero(np.array(mask) == 1)

            # token = config.tokenizer.tokenize(content)
            # token = [CLS] + token
            # seq_len = len(token)
            # mask = []
            # token_ids = config.tokenizer.convert_tokens_to_ids(token)

            # if pad_size:
            #     if len(token) < pad_size:
            #         mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            #         token_ids += ([0] * (pad_size - len(token)))
            #     else:
            #         # print(f"token size {len(token)} exceed pad_size {pad_size}")
            #         mask = [1] * pad_size
            #         token_ids = token_ids[-pad_size:]
            #         seq_len = pad_size
            contents.append((input_ids, int(label), seq_len, mask))

        return contents

    tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-cased',
        truncation_side='left',
        padding_side='right'
    )
    # tokenizer.pad_token_id = 50256
    data = pd.read_csv(config.train_path)
    # 划分数据集: 80%的数据用作训练集, 10%的数据用作验证集,10%的数据用作测试集
    train_set, intermediate = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    val_set, test_set = train_test_split(intermediate, test_size=0.5, random_state=42, shuffle=True)

    # 查看每个集体的大小
    print(f"Train set shape: {train_set.shape}")
    print(f"Validation set shape: {val_set.shape}")
    print(f"Test set shape: {test_set.shape}")
    # train_set = train_set.sample(1000)
    # val_set = val_set.sample(1000)
    # test_set = test_set.sample(1000)
    train = load_dataset(train_set, tokenizer, config.pad_size)
    dev = load_dataset(val_set, tokenizer, config.pad_size)
    test = load_dataset(test_set, tokenizer, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
