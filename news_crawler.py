
import sys, os
import requests
import selenium
from selenium import webdriver
import requests
from pandas import DataFrame
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pickle, progressbar, json, glob, time
from tqdm import tqdm

###### 날짜 저장 ##########
date = str(datetime.now())
date = date[:date.rfind(':')].replace(' ', '_')
date = date.replace(':','시') + '분'

sleep_sec = 0.5


####### 언론사별 본문 위치 태그 파싱 함수 ###########
print('본문 크롤링에 필요한 함수를 로딩하고 있습니다...\n' + '-' * 100)
def crawling_main_text(url):

    req = requests.get(url)
    req.encoding = None
    soup = BeautifulSoup(req.text, 'html.parser')
    
    try:
        # 연합뉴스
        if ('://yna' in url) | ('app.yonhapnews' in url): 
            main_article = soup.find('div', {'class':'story-news article'})
            if main_article == None:
                main_article = soup.find('div', {'class' : 'article-txt'})
            text = main_article.text
        # MBC 
        elif '//imnews.imbc' in url: 
            text = soup.find('div', {'itemprop' : 'articleBody'}).text
            
        # 매일경제(미라클), req.encoding = None 설정 필요
        elif 'mirakle.mk' in url:
            text = soup.find('div', {'class' : 'view_txt'}).text
            
        # 매일경제, req.encoding = None 설정 필요
        elif 'mk.co' in url:
            text = soup.find('div', {'class' : 'art_txt'}).text
            
        # SBS
        elif 'news.sbs' in url:
            text = soup.find('div', {'itemprop' : 'articleBody'}).text
        
        # KBS
        elif 'news.kbs' in url:
            text = soup.find('div', {'id' : 'cont_newstext'}).text
            
        # JTBC
        elif 'news.jtbc' in url:
            text = soup.find('div', {'class' : 'article_content'}).text
        else:
            raise Exception()
    except:
        return None
        
    return text.replace('\n','').replace('\r','').replace('<br>','').replace('\t','')
    
    
press_list = ['연합뉴스','KBS','매일경제','MBC','SBS','JTBC']

print('검색할 언론사 : {} | {}개 \n'.format(press_list, len(press_list)))


############### 브라우저를 켜고 검색 키워드 입력 ####################
query = input('검색할 키워드  : ')
news_num = int(input('수집 뉴스의 수(숫자만 입력) : '))

print('\n' + '=' * 100 + '\n')

print('브라우저를 실행시킵니다(자동 제어)\n')
chrome_path = '/usr/bin/chromedriver'
browser = webdriver.Chrome(chrome_path)

#'https://search.naver.com/search.naver?where=news&query=%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=2015.01.01&de=2015.12.31&docid=&nso=so%3Ar%2Cp%3Afrom20150101to20151231%2Ca%3Aall&mynews=0&refresh_start=0&related=0'#
news_url = 'https://search.naver.com/search.naver?where=news&query={}'.format(query)
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
browser.get(news_url,headers=headers)
time.sleep(sleep_sec)


######### 언론사 선택 및 confirm #####################
print('설정한 언론사를 선택합니다.\n')
search_opt_box = browser.find_element_by_xpath('//*[@id="search_option_button"]')
if search_opt_box.get_attribute('aria-pressed')=='false':
    search_opt_box.click()
time.sleep(0.02)

# 언론사 선택하는 바를 활성화
tablist_box = browser.find_element_by_xpath('//div[@class="snb_inner"]/ul[@role="tablist" and @class="option_menu"]')

tablist_elem_list = tablist_box.find_elements_by_xpath('./li[@role="presentation"]')
press_box = [t for t in tablist_elem_list if t.text == '언론사'][0].find_element_by_xpath('./a')
press_box.click()


# 언론사 종류 하나씩 선택
actived_press_frame = browser.find_element_by_xpath('.//div[@class="snb_itembox lst_press _search_option_press_"]')
total_press_box = actived_press_frame.find_element_by_xpath('./div[@class="group_sort type_press _group_by_press_"]')

# 언론사 종류를 선택하는 버튼이 담긴 박스
press_cat_active_button = total_press_box.find_elements_by_xpath('.//a[@role="tab" and @class="item _tab_filter_"]') # 언론사 종류 하나씩 버튼
press_cat_active_button_dict = dict(zip([t.text for t in press_cat_active_button], press_cat_active_button)) # 언론사 종류 이름 : 언론사 종류 활성화 버튼

# 밑에 각 언론사 종류별 개별 언론사가 담겨있는 박스들
each_press_box_list = total_press_box.find_elements_by_xpath('.//div[@class="scroll_area _panel_filter_"]')

# 1. 언론사 종류 1개 선택
# 2. 선택한 언론사 종류에 해당하는 개별 언론사 중 크롤링할 언론사에 포함되는 것 체크 
for idx, press_cat_name in enumerate(press_cat_active_button_dict.keys()):
    #하나의 언론사 종류를 클릭해서 활성화시킴
    press_cat_active_button_dict[press_cat_name].click()
    time.sleep(0.05)
    
    # 선택한 언론사 종류 안의 개별 언론사가 담긴 박스
    each_press_box = each_press_box_list[idx].find_element_by_xpath('./div[@class="select_item"]')
    # 개별 언론사의 이름
    each_press_title_list = [ep.get_attribute('title') for ep in each_press_box.find_elements_by_xpath('.//label')]
    # 개별 언론사 체크 박스
    each_press_input_list = each_press_box.find_elements_by_xpath('.//input')
    

    # 딕셔너리(개별 언론사 이름 : 개별 언론사 체크 박스)
    each_press_title_input_dict = dict(zip(each_press_title_list, each_press_input_list))
    # 추출하고 싶은 언론사 존재 시 체크박스 클릭
    for title in [tit for tit in each_press_title_input_dict.keys() if tit in press_list]:
        print(title)
        each_press_title_input_dict[title].click()


# 확인 버튼
confirm_buttons = actived_press_frame.find_element_by_xpath('./span[@class="btn_inp"]').find_elements_by_xpath('.//button')
ok_button = [c for c in confirm_buttons if c.text == '확인'][0]
ok_button.click()





################ 뉴스 크롤링 ########################

print('\n크롤링을 시작합니다.')
time.sleep(sleep_sec)
# ####동적 제어로 페이지 넘어가며 크롤링
news_dict = {}
idx = 1
cur_page = 1

pbar = tqdm(total=news_num)
    
while idx < news_num:
    print(123)
    table = browser.find_element_by_xpath('//ul[@class="list_news"]')
    li_list = table.find_elements_by_xpath('./li[contains(@id, "sp_nws")]')
    area_list = [li.find_element_by_xpath('.//div[@class="news_area"]') for li in li_list]
    a_list = [area.find_element_by_xpath('.//a[@class="news_tit"]') for area in area_list]

    for n in a_list[:min(len(a_list), news_num-idx+1)]:
        n_url = n.get_attribute('href')
        text = crawling_main_text(n_url)
        if text==None:
            continue
        news_dict[idx] = {'title' : n.get_attribute('title'), 
                          'url' : n_url,
                          'text' : text}
        idx += 1
        pbar.update(1)
        
    if idx < news_num:
        cur_page +=1

        pages = browser.find_element_by_xpath('//div[@class="sc_page_inner"]')
        next_page_url = [p for p in pages.find_elements_by_xpath('.//a') if p.text == str(cur_page)][0].get_attribute('href')

        browser.get(next_page_url)
        time.sleep(sleep_sec)
    else:
        pbar.close()
        
        print('\n브라우저를 종료합니다.\n' + '=' * 100)
        time.sleep(0.7)
        browser.close()
        break

#### 데이터 전처리하기 ###################################################### 

print('데이터프레임 변환\n')
news_df = DataFrame(news_dict).T

folder_path = os.getcwd()
xlsx_file_name = '네이버뉴스_본문_{}개_{}_{}.csv'.format(news_num, query, date)

news_df.to_csv(xlsx_file_name)

print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))

#os.startfile(folder_path)

print('=' * 100 + '\n결과물의 일부')
news_df