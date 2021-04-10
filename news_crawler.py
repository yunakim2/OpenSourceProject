#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


# Config
sleep_sec = 0.5


# In[ ]:


# Date Utility
def date_range(start_date, end_date):
    for n in range(int((end_date - start_date + timedelta(1)).days)):
        yield start_date + timedelta(n)
def parse_date(s):
    return datetime.strptime(s, '%Y.%m.%d')


# In[ ]:


# Press Crawler
press_list=['매일경제']
def crawling_main_text(url):
    req = requests.get(url)
    req.encoding = None
    soup = BeautifulSoup(req.text, 'html.parser')
    try:
        text = soup.find('div', {'class' : 'view_txt'}).text
    except:
        try:
            text = soup.find('div', {'class' : 'art_txt'}).text
        except:
            return None,None
    return text.replace('\n','').replace('\r','').replace('<br>','').replace('\t',''),soup.find('li',{'class':'lasttime'}).text

# In[ ]:


# Naver Crawler
keyword = input('Keyword: ')
news_num_per_day = int(input('crawl count per day: '))
date_start = parse_date(input('start date(YYYY.MM.DD): '))
date_end = parse_date(input('end date(YYYY.MM.DD): '))

driver_path = '/usr/bin/chromedriver'
browser = webdriver.Chrome(driver_path)

news_list=[]
for date in date_range(date_start,date_end):
    news_url = 'https://search.naver.com/search.naver?where=news&query={0}&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds={1}&de={1}'.format(keyword,date.strftime('%Y.%m.%d'))
    #headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    browser.get(news_url)#,headers=headers)
    time.sleep(sleep_sec)

    # 언론사 선택
    search_opt_box = browser.find_element_by_xpath('//*[@id="search_option_button"]')
    if search_opt_box.get_attribute('aria-pressed')=='false':
        search_opt_box.click()
    time.sleep(0.02)
    tablist_box = browser.find_element_by_xpath('//div[@class="snb_inner"]/ul[@role="tablist" and @class="option_menu"]')
    tablist_elem_list = tablist_box.find_elements_by_xpath('./li[@role="presentation"]')
    press_box = [t for t in tablist_elem_list if t.text == '언론사'][0].find_element_by_xpath('./a')
    press_box.click()
    actived_press_frame = browser.find_element_by_xpath('.//div[@class="snb_itembox lst_press _search_option_press_"]')
    total_press_box = actived_press_frame.find_element_by_xpath('./div[@class="group_sort type_press _group_by_press_"]')
    press_cat_active_button = total_press_box.find_elements_by_xpath('.//a[@role="tab" and @class="item _tab_filter_"]')
    press_cat_active_button_dict = dict(zip([t.text for t in press_cat_active_button], press_cat_active_button))
    each_press_box_list = total_press_box.find_elements_by_xpath('.//div[@class="scroll_area _panel_filter_"]')
    for idx, press_cat_name in enumerate(press_cat_active_button_dict.keys()):
        press_cat_active_button_dict[press_cat_name].click()
        time.sleep(0.05)
        each_press_box = each_press_box_list[idx].find_element_by_xpath('./div[@class="select_item"]')
        each_press_title_list = [ep.get_attribute('title') for ep in each_press_box.find_elements_by_xpath('.//label')]
        each_press_input_list = each_press_box.find_elements_by_xpath('.//input')
        each_press_title_input_dict = dict(zip(each_press_title_list, each_press_input_list))
        for title in [tit for tit in each_press_title_input_dict.keys() if tit in press_list]:
            each_press_title_input_dict[title].click()
    confirm_buttons = actived_press_frame.find_element_by_xpath('./span[@class="btn_inp"]').find_elements_by_xpath('.//button')
    ok_button = [c for c in confirm_buttons if c.text == '확인'][0]
    ok_button.click()
    
    print('Crawling news about {} on {}'.format(keyword,date.strftime('%Y.%m.%d')))
    time.sleep(sleep_sec)

    idx = 0
    cur_page = 1
    pbar = tqdm(total=news_num_per_day)
    try:
        while idx < news_num_per_day:
            table = browser.find_element_by_xpath('//ul[@class="list_news"]')
            li_list = table.find_elements_by_xpath('./li[contains(@id, "sp_nws")]')
            area_list = [li.find_element_by_xpath('.//div[@class="news_area"]') for li in li_list]
            a_list = [area.find_element_by_xpath('.//a[@class="news_tit"]') for area in area_list]

            for n in a_list[:min(len(a_list), news_num_per_day-idx)]:
                n_url = n.get_attribute('href')
                text,text_time = crawling_main_text(n_url)
                if text:
                    news_list.append({'title':n.get_attribute('title'),'url':n_url,'text':text,'date':date.strftime('%Y.%m.%d'),'time':text_time})
                    idx += 1
                    pbar.update(1)
            
            if idx < news_num_per_day:
                cur_page +=1
                pages = browser.find_element_by_xpath('//div[@class="sc_page_inner"]')
                next_page = [p for p in pages.find_elements_by_xpath('.//a') if p.text == str(cur_page)]
                if not next_page:
                    break
                next_page_url = next_page[0].get_attribute('href')
                browser.get(next_page_url)
                time.sleep(sleep_sec)
    except Exception as e:
        print('\n',e)
        pass
    pbar.close()
browser.close()
print('\nDone')


# In[ ]:


# Save
news_df = DataFrame(dict(enumerate(news_list))).T
folder_path = os.getcwd()
file_name = '{}_{}_{}_{}.csv'.format(keyword,date_start.strftime('%Y.%m.%d'),date_end.strftime('%Y.%m.%d'),news_num_per_day)
news_df.to_csv(file_name)
print('Saved at {}\\{}'.format(folder_path,file_name))

