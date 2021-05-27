import pandas as pd

if __name__ == '__main__':
    news = pd.read_csv('/Users/kim-yuna/PycharmProjects/NewsStockProject/data/test_data/all_processing_01.csv')
    news.drop(['Unnamed: 0'], axis=1, inplace=True)

    positive = 0
    negative = 0
    for l in news['label']:
        if l == 0:
            negative +=1
        else :
            positive +=1

    print('positive - ', positive)
    print('neg -', negative)
