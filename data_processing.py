# In[ ]:

import pandas as pd
import dateutil.parser

stock=pd.read_csv('data/stock/삼성전자2010.csv')
stock=stock.set_index('Date')
stock.index=pd.to_datetime(stock.index.map(dateutil.parser.parse)+pd.DateOffset(hours=15,minutes=30))

#결측값 제거
stock = stock[stock.index.dayofweek!=6]#이거 필요한가?
stock = stock.dropna(how='any',axis=0)
stock['Change'] = stock['Close'].astype('float').pct_change()

stock=stock[['Change']]
stock=stock.sort_values('Date')

#이동평균 적용
stock=stock.rolling('3d').mean()

news=pd.read_csv('data/news/삼성전자_2010.01.01_2021.04.12_3.csv')

#temp code(remove whitespaces)
#news=news.dropna()
#news['time']=news['time'].replace(u'(입력 :)|(수정 :.*)|(\xa0)|\n|\t','',regex=True).str.strip(' ')

news['time']=pd.to_datetime(news['time'].map(dateutil.parser.parse))
news=news.sort_values('time')

labeled_data = pd.merge_asof(news,stock,left_on='time',right_on='Date',direction='forward')

#labeled_data['text']=labeled_data['text'].str.split('.')
#labeled_data=labeled_data.explode('text')

labeled_data=labeled_data.drop(columns=['Unnamed: 0'])
labeled_data=labeled_data.rename(columns={'Change':'label'})

labeled_data.dropna().to_csv('data/labeled/samsung_2010_2021.csv')
# %%
