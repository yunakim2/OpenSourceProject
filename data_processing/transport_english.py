import pandas as pd
import numpy as np
import re
import googletrans
from googletrans import Translator
from google.cloud import translate_v2 as translate
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/kim-yuna/PycharmProjects/NewsStockProject/My Project 4833-29f2e885d35c.json'


if __name__ == '__main__':
    # translator = Translator()

    client = translate.Client()
    news = pd.read_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01_eng.csv')
    news.drop(['Unnamed: 0'], axis=1, inplace=True)

    idx = 0
    newss = news.copy()
    for txt in newss['title']:
        if idx >= 2033:
            print(txt)
            trans_txt = client.translate(txt, source_language='ko',target_language='en')
            newss.loc[idx, 'title'] = str(trans_txt['translatedText'])
        idx += 1

    newss.to_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01_eng.csv')