# In[ ]:

import sys, os
import requests
import selenium
from selenium import webdriver
import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime,timedelta,date
import pickle, progressbar, json, glob, time
from tqdm import tqdm

# Config
sleep_sec = 0.5

# Date Utility
def date_range(start_date, end_date):
    for n in range(int((end_date - start_date + timedelta(1)).days)):
        yield start_date + timedelta(n)
def parse_date(s):
    return datetime.strptime(s, '%Y.%m.%d')

# Article Crawler
def crawling_main_text(url):
    if 'news.naver.com' in url:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
        req = requests.get(url,headers=headers)
        req.encoding = None
        soup = BeautifulSoup(req.text, 'html.parser')
        try:
            text = soup.find('div', {'id' : 'dic_area'}).text
        except:
            return None,None
        text=text.replace('\n','').replace('\r','').replace('<br>','').replace('\t','')
        time_text=soup.find('span',{'class':'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME'})['data-date-time']
        return text,time_text
    else:
        return None,None

# Newslist Crawler
keyword = input('Keyword: ')
news_num_per_day = int(input('crawl count per day: '))
date_start = parse_date(input('start date(YYYY.MM.DD): '))
date_end = parse_date(input('end date(YYYY.MM.DD): '))

driver_path = '/usr/bin/chromedriver'
browser = webdriver.Chrome(driver_path)

news_list=[]
for date in date_range(date_start,date_end):
    news_url = 'https://m.search.naver.com/search.naver?where=m_news&query={0}&sm=mtb_tnw&sort=0&photo=0&field=0&pd=3&ds={1}&de={1}'.format(keyword,date.strftime('%Y.%m.%d'))
    browser.get(news_url)
    time.sleep(sleep_sec)

    idx = 0
    pbar = tqdm(total=news_num_per_day)
    try:
        while idx < news_num_per_day:
            news_ul = browser.find_element_by_xpath('//ul[@class="list_news"]')
            li_list = news_ul.find_elements_by_xpath('./li[@class="bx"]')
            item_list = [li.find_element_by_xpath('.//a[@class="news_tit"]') for li in li_list]

            for i in item_list[:min(len(item_list), news_num_per_day-idx)]:
                title = i.find_element_by_xpath('./div').text
                url = i.get_attribute('href')
                text,text_time = crawling_main_text(url)
                if text:
                    news_list.append({'title':title,'url':url,'text':text,'date':date.strftime('%Y.%m.%d'),'time':text_time})
                    idx += 1
                    pbar.update(1)
            
            if idx < news_num_per_day:
                button_next=browser.find_element_by_xpath('//button[@class="btn_next"]')
                if not button_next:
                  break
                button_next.click()
                time.sleep(sleep_sec)
    except Exception as e:
        print('\n',e)
        pass
    pbar.close()
browser.close()
print('\nDone')

# Save
news_df = DataFrame(dict(enumerate(news_list))).T
folder_path = os.getcwd()
file_name = '{}_{}_{}_{}.csv'.format(keyword,date_start.strftime('%Y.%m.%d'),date_end.strftime('%Y.%m.%d'),news_num_per_day)
news_df.to_csv(file_name)
print('Saved at {}/{}'.format(folder_path,file_name))
# %%
