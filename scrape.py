from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
from requests_html import HTMLSession

BASE_URL = 'https://strongmanarchives.com'
WSM_URL = 'https://strongmanarchives.com/contests.php?type=1'
ARNOLD_CLASSIC_URL = 'https://strongmanarchives.com/contests.php?type=7'
ARNOLD_CLASSIFIERS_URL = 'https://strongmanarchives.com/contests.php?type=18'
WUS_URL = 'https://strongmanarchives.com/contests.php?type=12'
GIANTS_URL = 'https://strongmanarchives.com/contests.php?type=5'
SHAW_CLASSIC_URL = 'https://strongmanarchives.com/contests.php?type=8'
ULTIMATE_STRONGMAN_URL = 'https://strongmanarchives.com/contests.php?type=30'
EUROPE_STRONGEST_MAN_URL = 'https://strongmanarchives.com/contests.php?type=9'
FORCA_BRUTA_URL = 'https://strongmanarchives.com/contests.php?type=10'
ROGUE_INVITATIONAL_URL = 'https://strongmanarchives.com/contests.php?type=68'
DATA_DIR = 'data/'


def load_page(url):
    driver = webdriver.Chrome(ChromeDriverManager().install())
    try:
        driver.get(url)
        time.sleep(1)

        text = driver.page_source
    except Exception as e:
        raise e
    # finally:
    #     driver.close()
    return BeautifulSoup(text, "lxml")


def load_competitions_page_selenium(url):
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

def load_page_v2(url):
    """Load competition page with results and event data, but without Selenium becouse it's slow."""

    session = HTMLSession()
    response = session.get(url)
    response.html.render(timeout=10, sleep=0.5)

    return BeautifulSoup(response.html.html, 'html.parser')


def parse_competition(competition_name, url, dir_name, is_wsm):
    """Parse and save a single competition (results + event data) to CSV"""

    page = load_page_v2(url)

    # Set paths
    results_data_dir_path = f'{DATA_DIR}/{dir_name}/results'
    event_data_dir_path = f'{DATA_DIR}/{dir_name}/events'

    # Parse with BS4
    results_data = parse_competition_results_data(page)
    events_data = parse_competition_events_data(page)

    # Save results and event data to CSVs
    save_to_csv(results_data, results_data_dir_path, competition_name, is_wsm)
    save_to_csv(events_data, event_data_dir_path, competition_name, is_wsm)


def parse_competition_results_data(page):
    """Parse results table to Pandas Dataframe with BS4"""

    tables = page.find_all('table', {
        'class': 'tablesorter tablesorter-blue tablesorter6bcc7b9997f32 hasFilters dataTable no-footer'})

    competition_data_dfs = []
    for table_id, table in enumerate(tables):
        headers = [th.text.strip() for th in table.find_all('th')]
        df = pd.DataFrame(columns=headers)
        rows = table.find_all('tr')[1:]
        if len(rows) > 0:
            for raw_row in rows:
                row_data = raw_row.find_all('td')

                row = [td.text.strip() for td in row_data]
                df.loc[len(df)] = row

            print(f'group {table_id}:', df)
            competition_data_dfs.append(df)

    return competition_data_dfs


def parse_competition_events_data(page):
    """Parse event data to Pandas Dataframe with BS4"""

    event_datas = page.find_all('div', {'class': 'content'})
    event_data_dfs = []

    for event_data in event_datas:
        lines = event_data.get_text(separator=" ").strip().split('\n')
        data = [line.split(' : ') for line in lines]
        if len(data) > 1:
            df = pd.DataFrame(columns=['Event', 'Info'], data=data)
            event_data_dfs.append(df)

    print(event_data_dfs)
    return event_data_dfs


def save_to_csv(data_dfs, dir_path, competition_name, is_wsm):
    """Create directories and save CSVs"""

    if not data_dfs:
        return

    # Create directories
    if is_wsm:
        finals_dir_path = f'{dir_path}/finals'
        groups_dir_path = f'{dir_path}/groups'
        create_dir(finals_dir_path)
        create_dir(groups_dir_path)
    else:
        create_dir(dir_path)

    # Save CSVs
    for table_id, table_df in enumerate(data_dfs):
        if is_wsm:
            if table_id == 0:
                table_df.to_csv(f'{dir_path}/finals/{competition_name}.csv', index=False)
            else:
                table_df.to_csv(f'{dir_path}/groups/{competition_name} - group {table_id}.csv', index=False)
        else:
            table_df.to_csv(f'{dir_path}/{competition_name}.csv', index=False)


def parse_all_competitions(url, dir_name, is_wsm):
    """Parse all competitions from a specified comp type ex. WSM, Arnold Classic etc."""

    page = load_competitions_page_selenium(url)
    table = page.find('table', id='CompTable')

    rows = table.find_all('tr')[1:]
    for raw_row in rows:
        row_data = raw_row.find_all('td')
        competition_name = row_data[1].text.strip()
        current_url = BASE_URL + [raw_row.find_all('a', href=True)][0][0]['href']

        parse_competition(competition_name, current_url, dir_name, is_wsm)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# parse_competition('2017 Arnold South America', 'https://strongmanarchives.com/viewContest.php?id=267', 'arnold_classifiers', is_wsm=False)

parse_all_competitions(WSM_URL, 'wsm', is_wsm=True)
