import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import os

BASE_URL = 'https://strongmanarchives.com'
URL = 'https://strongmanarchives.com/viewContest.php?id=825'
DATA_DIR = 'data/'
WSM_FINALS_DIR = 'data/wsm/finals/'
WSM_GROUPS_DIR = 'data/wsm/groups/'


def load_page(url):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    try:
        driver.get(url)
        text = driver.page_source
    except Exception as e:
        raise e
    finally:
        driver.close()
    return BeautifulSoup(text, "lxml")

def load_base_wsm_page(url):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    try:
        driver.get(url)
        select_fr = Select(driver.find_element(by='name', value='CompTable_length'))
        select_fr.select_by_value('100')

        text = driver.page_source
    except Exception as e:
        raise e
    finally:
        driver.close()
    return BeautifulSoup(text, "lxml")


def parse_wsm_page(date, competition_name, location, url):
    page = load_page(url)
    tables = page.find_all('table', {
        'class': 'tablesorter tablesorter-blue tablesorter6bcc7b9997f32 hasFilters dataTable no-footer'})

    table_id = 0
    for table in tables:
        headers = [th.text.strip() for th in table.find_all('th')]
        df = pd.DataFrame(columns=headers)
        rows = table.find_all('tr')[1:]
        if len(rows) > 0:
            for raw_row in rows:
                row_data = raw_row.find_all('td')

                row = [td.text.strip() for td in row_data]
                df.loc[len(df)] = row

            print(df)
            if table_id == 0:
                df.to_csv(f'{WSM_FINALS_DIR}{competition_name}.csv')
            else:
                df.to_csv(f'{WSM_GROUPS_DIR}{competition_name} - group {table_id}.csv')

            table_id += 1


def parse_all_wsm():
    wsm_url = 'https://strongmanarchives.com/contests.php?type=1'
    page = load_base_wsm_page(wsm_url)

    table = page.find('table', id='CompTable')

    rows = table.find_all('tr')[1:]
    for raw_row in rows:
        row_data = raw_row.find_all('td')
        date = row_data[0].text.strip()
        competition_name = row_data[1].text.strip()

        location = row_data[2].text.strip()
        print([raw_row.find_all('a', href=True)][0][0])
        current_wsm_url = BASE_URL + [raw_row.find_all('a', href=True)][0][0]['href']

        parse_wsm_page(date, competition_name, location, current_wsm_url)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# df.to_csv(f'{DATA_DIR}2021 wsm.csv')
parse_all_wsm()