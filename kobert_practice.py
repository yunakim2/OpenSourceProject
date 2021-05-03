# In[]
import gc
import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import numpy as np
import random
import time
import datetime

import io
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dsets

USE_CUDA = False
RANDOM_SEED=43 # 재현을 위해 랜덤시드 고정
TOKEN_MAX_LEN = 128*4
BATCH_SIZE = 2
STATUS_PRINT_INTERVAL=25

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

#%%
data = pd.read_csv('data/labeled/samsung_2010_2021.csv', encoding='utf-8', dtype={'label':np.float32})
test_cnt = int(data.shape[0] * 0.25)

test = data[:test_cnt]
train = data[test_cnt:]

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

# Train,Validation Data Preprocessing
input_ids = [tokenizer.encode(s,max_length=TOKEN_MAX_LEN,truncation=True) for s in train['text']]
input_ids = pad_sequences(input_ids, maxlen=TOKEN_MAX_LEN, dtype='long', truncating='post', padding='post')
attention_mask = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_mask.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, train['label'].values, random_state=RANDOM_SEED, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_mask,input_ids,random_state=RANDOM_SEED,test_size=0.1)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

# Test Data Preprocessing
input_ids = [tokenizer.encode(sent,max_length=TOKEN_MAX_LEN,truncation=True) for sent in test['text']]
input_ids = pad_sequences(input_ids, maxlen=TOKEN_MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(test['label'].values)
test_masks = torch.tensor(attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

if USE_CUDA and torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=1)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

model.zero_grad()

for epoch_i in range(0, epochs):
    print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step and step % STATUS_PRINT_INTERVAL == 0:
            elapsed = format_time(time.time() - t0)
            print('{:>5,}/{:>5,}, Elapsed {:}'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    avg_train_loss = total_loss / len(train_dataloader)

    print("\n  Average training loss: {0:.8f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    print("\nRunning Validation...")

    t0 = time.time()
    model.eval()
    eval_mae=0
    for batch in validation_dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
        pred = outputs[1].detach().cpu().numpy()
        real = b_labels.to('cpu').numpy()
        eval_mae += np.mean(np.abs(pred-real))
    print("  Validation MAE: {0:.8f}".format(eval_mae / len(validation_dataloader)))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("\nTraining complete!")

t0 = time.time()
model.eval()
eval_mae=0
for step, batch in enumerate(test_dataloader):
    if step and step % STATUS_PRINT_INTERVAL == 0:
        elapsed = format_time(time.time() - t0)
        print('{:>5,}/{:>5,}, Elapsed {:}'.format(step, len(test_dataloader), elapsed))
    b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
    pred = outputs[1].detach().cpu().numpy()
    real = b_labels.to('cpu').numpy()
    eval_mae += np.mean(np.abs(pred-real))
print("\nTest MAE: {0:.8f}".format(eval_mae / len(test_dataloader)))
print("Test took: {:}".format(format_time(time.time() - t0)))
print('TestSet AAD: {}'.format(test['label']))
# %%
