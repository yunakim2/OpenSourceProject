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
    """[CLS] [SEP] 문장으로 변환 (뉴스 내용)"""

    for idx in range(len(data)):
        document_bert = []
        split_text = data.iloc[idx]['text'].split(' ')
        for s in split_text:
            document_bert.append("[CLS]" + str(s) + "[SEP]")

        print(document_bert)
        data.loc[idx, 'text'] = ''.join(document_bert)

    return data['text']


def koberPreProcessing(test, train):
    """kobert - test, train 데이터 전처리"""

    train_document_bert = train['text']

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
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    """테스트 셋 전처리"""
    test_document_bert = test['text']
    tokenized_texts = [tokenizer.tokenize(sent) for sent in test_document_bert]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    test_inputs, validation_inputs, test_labels, validation_labels = \
        train_test_split(input_ids, train['text'].values, random_state=42, test_size=0.1)

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

    return test_dataloader, train_dataloader, validation_dataloader


def newsDataProcessing():
    """뉴스 데이터 전처리 25% - test, 75% - train"""
    data = pd.read_csv("data/kakao2020_processing_data", encoding='utf-8')
    test_cnt = int(data.shape[0] * 0.25)

    test = data[:test_cnt]
    train = data[test_cnt:]

    return test, train


def newsStockProcessing():
    """뉴스 데이터와 주식 변동량을 labeling 하는 작업"""

    data = pd.read_csv("data/카카오_2020.01.01_2020.12.31_3.csv", encoding='utf-8')
    '''news_data label 열 추가'''
    data['label'] = 0.0
    stock = pd.read_csv("data/stock/kakao2020.csv")
    stock = stock[::-1]
    i = 0
    for idx in range(len(stock)):
        print(idx, stock.iloc[idx]['날짜'], stock.iloc[idx]['변동 %'])
        while True:
            tmp_stock_date = stock.iloc[idx]['날짜']
            tmp_stock_date = tmp_stock_date.replace('년','').replace('월','').replace('일','')
            tmp_stock_date = tmp_stock_date.split(" ")
            stock_date = datetime.date(int(tmp_stock_date[0]), int(tmp_stock_date[1]), int(tmp_stock_date[2]))
            tmp_news_date = data.iloc[i]['date']
            tmp_news_date = tmp_news_date.split(".")
            news_date = datetime.date(int(tmp_news_date[0]),int(tmp_news_date[1]),int(tmp_news_date[2]))
            date_diff = stock_date - news_date
            print(date_diff.days ,news_date, data.iloc[i]['title'])
            if date_diff.days <= 0:
                break

            if date_diff.days == 1:
                data.loc[i, 'label'] = float(stock.iloc[idx]['변동 %'].replace('%',''))

            i += 1

    np_title = []
    np_url = []
    np_text = []
    np_date = []
    np_label = []

    for idx in range(len(data)):
        split_text = data.iloc[idx]['text'].split('.')
        for s in split_text:
            np_text.append("[CLS]" + str(s) + "[SEP]")
            np_title.append(data.iloc[idx]['title'])
            np_url.append(data.iloc[idx]['url'])
            np_date.append(data.iloc[idx]['date'])
            np_label.append(data.iloc[idx]['label'])

    data_df = pd.DataFrame({'title': np_title, 'url':np_url, 'text':np_text, 'date': np_date, 'label': np_label})

    file_name = "data/kakao2020_processing_data"
    data_df.to_csv(file_name)



def koBERTClassification(train_dataloader):
    """분류를 위한 BERT 모델 생성"""
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=1)
    model.cuda()

    """학습 스케줄링"""
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return epochs, scheduler, optimizer, model


def flat_accuracy(preds, labels):
    """학습 - accuracy"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """학습 -  시간표시 함수"""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def learningBERT(model, epochs, train_dataloader, test_dataloader, validation_dataloader, optimizer):
    """학습 실행 부분"""

    '''재현을 위해 랜덤시드 고정 '''
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    '''그래디언트 초기화'''
    model.zero_grad()

    '''애폭만큼 반복'''
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 시작 시간 설정
        t0 = time.time()

        # 로스 초기화
        total_loss = 0

        # 훈련모드로 변경
        model.train()

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(train_dataloader):
            # 경과 정보 표시
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            # 로스 구함
            loss = outputs[0]

            # 총 로스 계산
            total_loss += loss.item()

            # Backward 수행으로 그래디언트 계산
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 그래디언트를 통해 가중치 파라미터 업데이트
            optimizer.step()

            # 스케줄러로 학습률 감소
            scheduler.step()

            # 그래디언트 초기화
            model.zero_grad()

            # 평균 로스 계산
            avg_train_loss = total_loss / len(train_dataloader)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            # 시작 시간 설정
            t0 = time.time()

            # 평가모드로 변경
            model.eval()

            # 변수 초기화
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # 데이터로더에서 배치만큼 반복하여 가져옴
            for batch in validation_dataloader:
                # 배치를 GPU에 넣음
                batch = tuple(t.to(device) for t in batch)

                # 배치에서 데이터 추출
                b_input_ids, b_input_mask, b_labels = batch

                # 그래디언트 계산 안함
                with torch.no_grad():
                    # Forward 수행
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

                # 로스 구함
                logits = outputs[0]

                # CPU로 데이터 이동
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # 출력 로짓과 라벨을 비교하여 정확도 계산
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")


if __name__ == '__main__':
    '''뉴스 데이터 전처러'''
    # newsStockProcessing()
    test, train = newsDataProcessing()
    test_dataloader, train_dataloader, validation_dataloader = koberPreProcessing(test, train)
    epochs, scheduler, optimizer, model = koBERTClassification(train_dataloader)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    learningBERT(model, epochs, train_dataloader, test_dataloader, validation_dataloader, optimizer)
