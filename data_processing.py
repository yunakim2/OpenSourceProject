# In[ ]:

import pandas as pd
import dateutil.parser

stock=pd.read_csv('data/stock/kakao_test.csv')
stock=stock.set_index('Date')
stock.index=pd.to_datetime(stock.index.map(dateutil.parser.parse)+pd.DateOffset(hours=15,minutes=30))

#결측값 제거
stock = stock.dropna(how='any',axis=0)
stock['Change'] = stock['Close'].astype('float').pct_change()

stock=stock[['Change']]
stock=stock.sort_values('Date')

news=pd.read_csv('data/news/kakao.csv')
news['time']=pd.to_datetime(news['time'].map(dateutil.parser.parse))
news=news.sort_values('time')

news['text']=news['text'].str.replace(input('keyword'),'종목명')
labeled_data = pd.merge_asof(news,stock,left_on='time',right_on='Date',direction='forward')

#labeled_data['text']=labeled_data['text'].str.split('.')
#labeled_data=labeled_data.explode('text')

labeled_data=labeled_data.drop(columns=['Unnamed: 0'])
labeled_data=labeled_data.rename(columns={'Change':'label'})

labeled_data.dropna().to_csv('data/labeled/kakao.csv')
# %%
