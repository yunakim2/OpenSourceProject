# In[ ]:

import pandas as pd

stock=pd.read_csv('data/stock/kakao2020.csv')
stock=stock.set_index('날짜')
stock.index=pd.to_datetime(stock.index,format='%Y년 %m월 %d일')+pd.DateOffset(hours=15,minutes=30)

#결측값 제거
stock = stock[stock.index.dayofweek!=6]
stock = stock[stock['거래량']!='-']
stock['변동 %']=stock['변동 %'].str.rstrip('%').astype('float32')/100

stock=stock[['변동 %']]
stock=stock[::-1]

# #결측값 공간 메우기 (for 이동평균)
# idx = pd.date_range('2020-01-01', '2020-12-31')
# stock = stock.set_index('날짜').reindex(idx,fill_value='0%').rename_axis('날짜').reset_index()
# print (stock[-20:])

#이동평균 적용
stock=stock.rolling('3d').mean()

news=pd.read_csv('data/news/카카오_2020.01.01_2020.12.31_3.csv')

#temp code(remove whitespaces)
news=news.dropna()
news['time']=news['time'].replace(u'(입력 :)|(수정 :.*)|(\xa0)|\n|\t','',regex=True).str.strip(' ')

import dateutil.parser
news['time']=pd.to_datetime(news['time'].map(dateutil.parser.parse))
news=news.sort_values('time')

labeled_data = pd.merge_asof(news,stock,left_on='time',right_on='날짜',direction='forward')

#labeled_data['text']=labeled_data['text'].str.split('.')
#labeled_data=labeled_data.explode('text')

labeled_data=labeled_data.drop(columns=['Unnamed: 0'])
labeled_data=labeled_data.rename(columns={'변동 %':'label'})

labeled_data.dropna().to_csv('data/labeled/kakao2020.csv')
