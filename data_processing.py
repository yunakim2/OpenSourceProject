import pandas as pd
import datetime

"""뉴스 데이터와 주식 데이터 날짜별로 주식변동량 추가 후 , 저장"""


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
            tmp_stock_date = tmp_stock_date.replace('년', '').replace('월', '').replace('일', '')
            tmp_stock_date = tmp_stock_date.split(" ")
            stock_date = datetime.date(int(tmp_stock_date[0]), int(tmp_stock_date[1]), int(tmp_stock_date[2]))
            tmp_news_date = data.iloc[i]['date']
            tmp_news_date = tmp_news_date.split(".")
            news_date = datetime.date(int(tmp_news_date[0]), int(tmp_news_date[1]), int(tmp_news_date[2]))
            date_diff = stock_date - news_date
            print(date_diff.days, news_date, data.iloc[i]['title'])
            if date_diff.days <= 0:
                break

            if date_diff.days == 1:
                data.loc[i, 'label'] = float(stock.iloc[idx]['변동 %'].replace('%', ''))

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

    data_df = pd.DataFrame({'title': np_title, 'url': np_url, 'text': np_text, 'date': np_date, 'label': np_label})

    file_name = "data/kakao2020_processing_data.csv"
    data_df.to_csv(file_name)


if __name__ == '__main__':
    newsStockProcessing()
