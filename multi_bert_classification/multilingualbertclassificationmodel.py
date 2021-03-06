#!/usr/bin/env python
# coding: utf-8

# In[1]:

#
# get_ipython().system('pip install transformers')
# get_ipython().system('pip install keras')
# get_ipython().system('pip install tensorflow')
# get_ipython().system('pip install jupyter-resource-usage')
# get_ipython().system('pip install mxnet')
# get_ipython().system('pip install gluonnlp pandas tqdm')
# get_ipython().system('pip install sentencepiece')
# get_ipython().system('pip install git+https://git@github.com/SKTBrain/KoBERT.git@master')
#
#
# # In[17]:
#
#
# # !wget https://raw.githubusercontent.com/yunakim2/OpenSourceProject/feat/bertModel/data/test_data/all_processing_01_eng.csv
# get_ipython().system('wget https://raw.githubusercontent.com/yunakim2/OpenSourceProject/feat/bertModel/data/test_data/all_processing_01.csv')

import gc
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


# In[18]:


import os

n_devices = torch.cuda.device_count()
print(n_devices)

for i in range(n_devices):
    print(torch.cuda.get_device_name(i))


# In[19]:


data = pd.read_csv('all_processing_01.csv',encoding = 'utf-8')
test_cnt = int(data.shape[0] * 0.25)

test = data[:test_cnt]
train = data[test_cnt:]
document_bert = ["[CLS] " + str(s) + " [SEP]" for s in train['text']]
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

tokenized_texts = [tokenizer.tokenize(s) for s in document_bert]

MAX_LEN = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
attention_masks = []

for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
    

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, train['label'].values, random_state=42, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                       input_ids,
                                                       random_state=42, 
                                                       test_size=0.1)
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)



# In[20]:


BATCH_SIZE = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)


# In[21]:


sentences = test['text']
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
labels = test['label'].values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


# In[22]:


model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
model.cuda()


# In[23]:


# ??????????????? ??????
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # ?????????
                  eps = 1e-8 # 0?????? ????????? ?????? ???????????? ?????? epsilon ???
                )

# ?????????
epochs = 4

# ??? ?????? ??????
total_steps = len(train_dataloader) * epochs

# lr ????????? ??????????????? ????????????
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


# In[24]:


# ????????? ?????? ??????
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# ?????? ?????? ??????
def format_time(elapsed):
    # ?????????
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss?????? ?????? ??????
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[25]:


# ????????? ?????? ???????????? ??????
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# ??????????????? ?????????
model.zero_grad()
device = "cuda:0"
model = model.to(device)
# ???????????? ??????
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # ?????? ?????? ??????
    t0 = time.time()

    # ?????? ?????????
    total_loss = 0

    # ??????????????? ??????
    model.train()
        
    # ????????????????????? ???????????? ???????????? ?????????
    for step, batch in enumerate(train_dataloader):
        # ?????? ?????? ??????
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # ????????? GPU??? ??????
        batch = tuple(t.to(device) for t in batch)
        
        # ???????????? ????????? ??????
        b_input_ids, b_input_mask, b_labels = batch

        # Forward ??????                
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        # ?????? ??????
        loss = outputs[0]

        # ??? ?????? ??????
        total_loss += loss.item()

        # Backward ???????????? ??????????????? ??????
        loss.backward()

        # ??????????????? ?????????
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ?????????????????? ?????? ????????? ???????????? ????????????
        optimizer.step()

        # ??????????????? ????????? ??????
        scheduler.step()

        # ??????????????? ?????????
        model.zero_grad()

    # ?????? ?????? ??????
    avg_train_loss = total_loss / len(train_dataloader)            

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    #?????? ?????? ??????
    t0 = time.time()

    # ??????????????? ??????
    model.eval()

    # ?????? ?????????
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # ????????????????????? ???????????? ???????????? ?????????
    for batch in validation_dataloader:
        # ????????? GPU??? ??????
        batch = tuple(t.to(device) for t in batch)
        
        # ???????????? ????????? ??????
        b_input_ids, b_input_mask, b_labels = batch
        
        # ??????????????? ?????? ??????
        with torch.no_grad():     
            # Forward ??????
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # ?????? ??????
        logits = outputs[0]

        # CPU??? ????????? ??????
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # ?????? ????????? ????????? ???????????? ????????? ??????
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")


# In[26]:


#?????? ?????? ??????
t0 = time.time()

# ??????????????? ??????
model.eval()


# ?????? ?????????
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

# ????????????????????? ???????????? ???????????? ?????????
for step, batch in enumerate(test_dataloader):
    # ?????? ?????? ??????
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # ????????? GPU??? ??????
    batch = tuple(t.to(device) for t in batch)
    
    # ???????????? ????????? ??????
    b_input_ids, b_input_mask, b_labels = batch
    
    # ??????????????? ?????? ??????
    with torch.no_grad():     
        # Forward ??????
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
    
    # ?????? ??????
    logits = outputs[0]

    # CPU??? ????????? ??????
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # ?????? ????????? ????????? ???????????? ????????? ??????
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))

