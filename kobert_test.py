import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

import os


def makeDocumentBert(data):
    document_bert = []
    for content in data['text']:
        split_text = content.split('.')
        for s in split_text:
            document_bert.append("[CLS]" + str(s) + "[SEP]")

    print(document_bert[:5])
    return document_bert


def kobertTestPreProcessing_(test, train):
    train_document_bert = makeDocumentBert(train)

    '''사전 학습된 BERT multilingual 모델 내 포함되어있는 토크나이저를 활용하여 토크나이징 함'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(s) for s in train_document_bert]
    print(tokenized_texts[0])

    '''패딩 과정'''
    MAX_LEN = 128
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

    '''어텐션 마스크'''
    '''실 데이터가 있는 곳과 padding이 있는 곳 attention에게 알려줌'''
    attention_mask = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_mask.append(seq_mask)

    '''trian - validation set 분리'''
    train_inputs, validation_inputs, train_labels, validation_labels = \
        train_test_split(input_ids, train['label'].values, random_state=42, test_size=0.1)

    train_masks, validation_masks, _, _ = train_test_split(attention_mask,
                                                           input_ids,
                                                           random_state=42,
                                                           test_size=0.1)


    '''파이토치 텐서로 변환'''
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    '''배치 및 데이터로더 설정'''
    BATCH_SIZE = 32
    train_data = TensorDataset(train_inputs,train_masks,train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,sampler=train_sampler, batch_size=BATCH_SIZE)

    validation_data = TensorDataset(validation_inputs,validation_masks,validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)


    '''테스트 셋 전처리'''
    test_document_bert = makeDocumentBert(test)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in test_document_bert]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    '''trian - validation set 분리'''
    test_inputs, validation_inputs, test_labels, validation_labels = \
        train_test_split(input_ids, train['label'].values, random_state=42, test_size=0.1)

    train_masks, validation_masks, _, _ = train_test_split(attention_mask,
                                                           input_ids,
                                                           random_state=42,
                                                           test_size=0.1)

    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


def newsDataProcessing():
    data = pd.read_csv("data/카카오_2020.01.01_2020.12.31_3.csv", encoding='utf-8')
    test_cnt = int(data.shape[0] * 0.25)
    train_cnt = data.shape[0] - test_cnt
    print(test_cnt)
    print(train_cnt)

    test = data[:test_cnt]
    train = data[test_cnt:]

    kobertTestPreProcessing_(test, train)


if __name__ == '__main__':
    # newsDataProcessing()
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print('There are %d GPU(s) available.' % torch.cuda.device_count())
    #     print('We will use the GPU:', torch.cuda.get_device_name(0))
    # else:
    #     device = torch.device("cpu")
    #     print('No GPU available, using the CPU instead.')
