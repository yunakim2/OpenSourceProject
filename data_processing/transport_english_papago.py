import pandas as pd
import os
import sys
import urllib.request
import requests
client_id = '1YQaJpNhfiWi0IPBijg0'
client_secret = '1MPlraITxz'




if __name__ == '__main__':
    # translator = Translator()

    news = pd.read_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01_eng.csv')
    news.drop(['Unnamed: 0'], axis=1, inplace=True)
    url = "https://openapi.naver.com/v1/papago/n2mt"
    idx = 0
    boolean = True
    newss = news.copy()
    for txt in newss['title']:
        if idx >= 2033:
            data = {'text': txt,
                    'source': 'ko',
                    'target': 'en'}

            header = {"X-Naver-Client-Id": client_id,
                      "X-Naver-Client-Secret": client_secret}

            response = requests.post(url, headers=header, data=data)
            rescode = response.status_code
            if (rescode == 200):
                response_body = response.json()
                newss.loc[idx, 'title'] = str(response_body['message']['result']['translatedText'])
            else:
                boolean = False
                break
        if not boolean:
            print(idx)
            break
        idx += 1


    newss.to_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01_eng.csv')