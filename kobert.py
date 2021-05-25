#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# !pip3 install transformers
# !pip3 install keras
# !pip3 install tensorflow
# !pip3 install jupyter-resource-usage
# !pip3 install mxnet
# !pip3 install gluonnlp pandas tqdm
# !pip3 install sentencepiece
# !pip3 install git+https://git@github.com/SKTBrain/KoBERT.git@master
# !wget https://raw.githubusercontent.com/yunakim2/OpenSourceProject/feat/kobart/data/labeled/all.csv


# In[ ]:


import gc
import torch

from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import gluonnlp as nlp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import numpy as np
import random
import time
import datetime

import io
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as dsets

USE_CUDA = True
RANDOM_SEED=43 # 재현을 위해 랜덤시드 고정
TOKEN_MAX_LEN = 128*4
BATCH_SIZE = 12 #로컬 6GB일때 1, 클라우드T4 16GB일때 12
STATUS_PRINT_INTERVAL=25
epochs = 10
LEARNING_RATE=1e-4

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

if USE_CUDA and torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

bertmodel, vocab = get_pytorch_kobert_model()


# In[ ]:


from transformers import BertPreTrainedModel,BertModel
class BertRegression(BertPreTrainedModel):
    def __init__(self, config):
        config.num_labels=1
        super().__init__(config)
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_bertkospi = nn.Linear(config.hidden_size+1, 128)
        self.linear_h1 = nn.Linear(128, 32)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(32, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        input_kospi=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        x = self.linear_bertkospi(torch.cat([pooled_output,input_kospi],dim=1))
        x = self.activation(x)
        x = self.linear_h1(x)
        x = self.activation(x)
        logits = self.classifier(x)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return logits
model = BertRegression.from_pretrained(bertmodel.config.name_or_path).to(device)
print('Model Created')


# In[ ]:


data = pd.read_csv('all.csv', encoding='utf-8', dtype={'label':np.float32,'ChangeK':np.float32})
test_cnt = int(data.shape[0] * 0.25)

test = data[:test_cnt]
train = data[test_cnt:]

tok = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
tokenizer = nlp.data.BERTSentenceTransform(tok,max_seq_length=TOKEN_MAX_LEN,vocab=vocab,pair=False)

print('train&validation data processing')
input_ids = np.array([tokenizer([i])[0] for i in train['text']]).astype(int)

print(input_ids)
print(vocab.to_tokens([int(i) for i in input_ids[0][:10]]))
attention_mask = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_mask.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, train['label'].values, random_state=RANDOM_SEED, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_mask,input_ids,random_state=RANDOM_SEED,test_size=0.1)
train_kospi, validation_kospi, _, _ = train_test_split(train['ChangeK'].values,input_ids,random_state=RANDOM_SEED,test_size=0.1)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_kospi = torch.tensor(train_kospi)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
validation_kospi = torch.tensor(validation_kospi)

train_data = TensorDataset(train_inputs, train_kospi, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_kospi, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

print('test data processing')
input_ids = np.array([tokenizer([i])[0] for i in test['text']]).astype(int)
print(vocab.to_tokens([int(i) for i in input_ids[0][:10]]))
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(test['label'].values)
test_masks = torch.tensor(attention_masks)
test_kospi = torch.tensor(test['ChangeK'].values)

test_data = TensorDataset(test_inputs, test_kospi, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_func = nn.MSELoss()
mae_func = lambda pred,label: torch.mean(torch.abs(pred-label)).item()

model.zero_grad()
for epoch_i in range(0, epochs):
    print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_mae = 0
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    for step, batch in enumerate(train_dataloader):
        pass
        if step and step % STATUS_PRINT_INTERVAL == 0:
            elapsed = format_time(time.time() - t0)
            print('{:>5,}/{:>5,}, Elapsed {:}'.format(step, len(train_dataloader), elapsed))
            print((pred.view(-1)*b_labels.view(-1)>=0).float().mean())

        b_input_ids, b_kospi, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        b_kospi=b_kospi.unsqueeze(dim=1)
        pred = model(b_input_ids, b_kospi, token_type_ids=None, attention_mask=b_input_mask)
        loss = loss_func(pred.view(-1), b_labels.view(-1))
        total_mae += mae_func(pred.view(-1), b_labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    print("\n  Training mae: {0:.8f}".format(total_mae / len(train_dataloader)))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    print("\nRunning Validation...")

    t0 = time.time()
    model.eval()
    total_mae=0
    for batch in validation_dataloader:
        b_input_ids, b_kospi, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        b_kospi=b_kospi.unsqueeze(dim=1)
        with torch.no_grad():
            pred = model(b_input_ids, b_kospi, token_type_ids=None, attention_mask=b_input_mask)
            loss = loss_func(pred.view(-1), b_labels.view(-1))
            total_mae += mae_func(pred.view(-1), b_labels.view(-1))
    print("  Validation MAE: {0:.8f}".format(total_mae / len(validation_dataloader)))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("\nTraining complete!")


# In[ ]:


print('TestSet MAE: {:.8f}'.format(mae_func(torch.tensor(test['label']),torch.tensor(test['label'].median()))))

t0 = time.time()
model.eval()
total_mae=0
acc=[]
for step, batch in enumerate(test_dataloader):
    if step and step % STATUS_PRINT_INTERVAL == 0:
        elapsed = format_time(time.time() - t0)
        print('{:>5,}/{:>5,}, Elapsed {:}'.format(step, len(test_dataloader), elapsed))
        #print((pred.view(-1)*b_labels.view(-1)>=0).float().mean())
    b_input_ids, b_kospi, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
    b_kospi=b_kospi.unsqueeze(dim=1)
    with torch.no_grad():
        pred = model(b_input_ids, b_kospi, token_type_ids=None, attention_mask=b_input_mask)
        loss = loss_func(pred.view(-1), b_labels.view(-1))
        total_mae += mae_func(pred.view(-1), b_labels.view(-1))
        acc+=(pred.view(-1)*b_labels.view(-1)>=0)
print("\nTest MAE: {0:.8f}".format(total_mae / len(test_dataloader)))
print("\nTest Acc: {0:.8f}".format(sum(acc) / len(acc)))
print("Test took: {:}".format(format_time(time.time() - t0)))
# %%

