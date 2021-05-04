import pandas as pd
import numpy as np
import re

if __name__ == '__main__':
    news = pd.read_csv('data/labeled/samsung_2010_2021.csv')
    news.drop(['Unnamed: 0'], axis=1, inplace=True)

    idx = 0
    newss = news.copy()
    for txt in newss['text']:
        txt = re.sub(r'[^\w]', ' ', txt) + ''
        txt = txt.replace("원본보기", '')
        txt = txt.strip()
        newss.loc[idx, 'text'] = txt
        idx += 1

    newss['label'].astype(int)
    newss['label'] = newss['label'].apply(lambda x: 1 if x >= 0 else 0)

    newss.to_csv('data/labeled/samsung_2010_2021_01.csv')



