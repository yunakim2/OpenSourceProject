# In[ ]:
import os
import pandas as pd
import dateutil.parser

kospi=pd.read_csv('data/stock/kospi.csv')
kospi=pd.DataFrame(kospi[['Date','Close']])
kospi=kospi.rename(columns={'Close':'CloseK'})
kospi=kospi.set_index('Date')

for keyword in os.listdir('data/news'):
	keyword=keyword.split('.')[0]

	stock=pd.read_csv('data/stock/{}.csv'.format(keyword))

	#TODO: Merge stock and kospi by date

	stock=stock.set_index('Date')
	stock=stock.join(kospi)

	stock.index=pd.to_datetime(stock.index.map(dateutil.parser.parse)+pd.DateOffset(hours=15,minutes=30))

	#결측값 제거
	stock = stock.dropna(how='any',axis=0)
	
	stock['Change'] = stock['Close'].astype('float').pct_change()
	stock['ChangeK'] = stock['CloseK'].astype('float').pct_change()

	stock=stock[['Change','ChangeK']]
	stock=stock.sort_values('Date')

	news=pd.read_csv('data/news/{}.csv'.format(keyword))
	news['time']=pd.to_datetime(news['time'].map(dateutil.parser.parse))
	news=news.sort_values('time')


	news.drop(['text','url'],inplace=True,axis=1,errors='ignore')
	#news['text']=news['text'].str.replace(keyword,'종목명')
	news['title']=news['title'].str.replace(keyword,'종목명')

	labeled_data = pd.merge_asof(news,stock,left_on='time',right_on='Date',direction='forward')

	#labeled_data['text']=labeled_data['text'].str.split('.')
	#labeled_data=labeled_data.explode('text')

	labeled_data=labeled_data.drop(columns=['Unnamed: 0'])
	labeled_data=labeled_data.rename(columns={'Change':'label'})

	labeled_data.dropna().to_csv('data/labeled/{}.csv'.format(keyword))
	# %%

all=pd.concat([pd.read_csv('data/labeled/{}.csv'.format(name)) for name in [keyword.split('.')[0] for keyword in os.listdir('data/labeled')]])
all.to_csv('data/labeled/all.csv')
# %%
