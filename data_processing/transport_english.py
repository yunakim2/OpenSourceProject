import pandas as pd
import numpy as np
import re
import googletrans
from googletrans import Translator
from google.cloud import translate_v2 as translate
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/kim-yuna/PycharmProjects/NewsStockProject/opensource-314902-67f557140926.json'


if __name__ == '__main__':
    # translator = Translator()
    client = translate.Client()
    news = pd.read_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01.csv')
    news.drop(['Unnamed: 0'], axis=1, inplace=True)

    idx = 0
    newss = news.copy()
    for txt in newss['text']:
        print(txt)
        trans_txt = client.translate(txt, source_language='ko',target_language='en')
        newss.loc[idx, 'text'] = str(trans_txt['translatedText'])
        idx += 1

    newss.to_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01_eng.csv')