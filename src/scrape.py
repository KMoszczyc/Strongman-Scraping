import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
from requests_html import HTMLSession
import html
import re
from collections import Counter
import math
import utils
from pathlib import Path
from ftfy import fix_encoding

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', 200)

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

ROOT_PATH = str(Path(os.path.abspath('')).parents[0])
DATA_RAW_DIR_PATH = os.path.join(ROOT_PATH, 'data/data_raw')
DATA_TRANSFORMED_DIR_PATH = os.path.join(ROOT_PATH, 'data/data_transformed')

print(DATA_RAW_DIR_PATH)

def load_page(url):
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    driver = webdriver.Chrome()

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
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    driver = webdriver.Chrome()
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
    """Load competition page with results and event data, but without Selenium because it's slow."""

    session = HTMLSession()
    response = session.get(url)
    response.html.render(timeout=10, sleep=0.5)
    # print(response)
    decoded_html = decode_html(response.html.html)
    # print(decoded_html)
    soup = BeautifulSoup(decoded_html, 'html.parser')

    return soup

def decode_html(html):
    """Repair special characters, some like: Á and á are replaced with A and a"""
    # return html.replace("\u00C3\uFFFD", "A").replace("\u00E1\uFFFD", 'a').replace("\uFFFD", '?').encode('windows-1252').decode('utf-8')
    return fix_encoding(html).replace("�", "Á")

def parse_competition(competition_name, url, dir_name, is_wsm, force_update=False):
    """Parse and save a single competition (results + event data) to CSV"""

    page = load_page_v2(url)

    # Set paths
    results_data_dir_path = f'{DATA_RAW_DIR_PATH}/{dir_name}/results'
    event_data_dir_path = f'{DATA_RAW_DIR_PATH}/{dir_name}/events'

    # Stop scraping if competition file is already downloaded and force_update flag is turned off.
    if (file_exists(results_data_dir_path, competition_name, is_wsm) and file_exists(event_data_dir_path, competition_name, is_wsm)) and not force_update:
        print('Scraping stopped! Competition file of:', competition_name, ' is already downloaded.')
        return

    # Parse with BS4
    results_data = parse_competition_results_data(page)
    events_data = parse_competition_events_data(page)

    # Save results and event data to CSVs
    save_to_csv(results_data, results_data_dir_path, competition_name, is_wsm)
    save_to_csv(events_data, event_data_dir_path, competition_name, is_wsm)

    print('Scraped:', competition_name)

def parse_competition_results_data(page):
    """Parse results table to Pandas Dataframe with BS4"""

    tables = page.find_all('table', {
        'class': 'tablesorter tablesorter-blue tablesorter6bcc7b9997f32 hasFilters dataTable no-footer'})

    # print(page)
    # print(tables)
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

            # print(f'group {table_id}:', df)
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
                table_df.to_csv(f'{dir_path}/finals/{competition_name}.csv', index=False, encoding='utf-8-sig')
            else:
                table_df.to_csv(f'{dir_path}/groups/{competition_name} - group {table_id}.csv', index=False, encoding='utf-8-sig')
        else:
            table_df.to_csv(f'{dir_path}/{competition_name}.csv', index=False)

def file_exists(dir_path, competition_name, is_wsm):
    path = f'{dir_path}/{competition_name}.csv'
    if is_wsm:
        path = f'{dir_path}/finals/{competition_name}.csv'

    return os.path.isfile(path)

def parse_all_competitions(url, dir_name, is_wsm, force_update=False):
    """Parse all competitions from a specified comp type ex. WSM, Arnold Classic etc."""

    page = load_competitions_page_selenium(url)
    table = page.find('table', id='CompTable')

    rows = table.find_all('tr')[1:]
    for raw_row in rows:
        row_data = raw_row.find_all('td')
        competition_name = row_data[1].text.strip()
        current_url = BASE_URL + [raw_row.find_all('a', href=True)][0][0]['href']

        parse_competition(competition_name, current_url, dir_name, is_wsm, force_update)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def transform_csv(results_path, events_path):
    info_schema = {
        'event_name': '',
        'distance': '',
        'time_limit': '',
        'main_measurement_unit': '',
        'second_measurement_unit': '',
        'min_weight': '',
        'max_weight': '',
        'weights': '',
        'deadlift': '',
        'attempts': '',
        'max_lifts': '',
        'lifts_num': '',
        'implements_num': '',
        'implements': '',
    }

    event_info_df = pd.read_csv(events_path, sep=',')
    results_df = pd.read_csv(results_path, sep=',')

    # print(results_df)
    # print(event_info_df)

    raw_event_results_df, raw_event_points_df, competitors = split_results(results_df)

    transformed_event_info, transformed_event_info_df = transform_event_info(raw_event_results_df, event_info_df, info_schema)
    transformed_result_points_df, transformed_result_units_df, transformed_result_values_df, transformed_event_info_df = transform_results(raw_event_results_df, raw_event_points_df,
                                                                                                                transformed_event_info, competitors)

    transformed_result_points_df = prepapre_df_for_merging(transformed_result_points_df, events_path, results_path)
    transformed_result_units_df = prepapre_df_for_merging(transformed_result_units_df, events_path, results_path)
    transformed_result_values_df = prepapre_df_for_merging(transformed_result_values_df, events_path, results_path)
    transformed_event_info_df = prepapre_df_for_merging(transformed_event_info_df, events_path, results_path)
    transformed_raw_results_df = prepapre_df_for_merging(raw_event_results_df, events_path, results_path)

    # print('----------------------Final---------------------')
    # print(transformed_result_points_df)
    # print(transformed_result_units_df)
    # print(transformed_result_values_df)
    # print(transformed_event_info_df)
    # print(transformed_raw_results_df)

    return transformed_result_points_df, transformed_result_units_df, transformed_result_values_df, transformed_event_info_df, transformed_raw_results_df


def transform_event_info(event_results_df, event_info_df, info_schema):
    # Preprocess event info
    event_info_df['Event'] = event_info_df['Event'].str.strip()
    event_info_df['Info'] = event_info_df['Info'].str.strip()

    # Reorder event info
    events_ordered = [event for event in list(event_results_df.columns) if event in list(event_info_df['Event'])]
    event_info_df = event_info_df.set_index('Event', drop=False)
    event_info_df = event_info_df.loc[events_ordered]

    transformed_event_info = []
    for i, row in event_info_df.iterrows():
        event_name = row['Event'].strip()
        info_dict = info_schema.copy()
        info_dict['event_name'] = event_name
        event_name_lower = event_name.lower()
        info_split = row['Info'].strip('\"').split('/')

        for info in info_split:
            info = remove_braces(info).strip()
            info_lower = info.lower()
            # results_with_in = [s for s in list(results_df[event_name]) if ' in ' in s]

            if 'second time limit' in info:
                info_dict['time_limit'] = get_int_before_word(info, 'second time limit')
            if 'meters' in info or 'metres' in info:
                info_dict['distance'] = get_float_before_word(info, 'meters')
            if ',' in info:
                info_dict['implements'] = get_implements(',', info)
            if '+' in info:
                info_dict['implements'] = get_implements('+', info)
            if info_dict['implements']:
                info_dict['implements_num'] = len(info_dict['implements'])
            if 'implements' in info:
                info_dict['implements_num'] = get_int_before_word(info, 'implements')
                info_dict['max_lifts'] = info_dict['implements_num']
            if 'lifts' in info:
                info_dict['max_lifts'] = get_int_before_word(info, 'lifts')
            if 'stairs' in info:
                info_dict['max_lifts'] = get_int_before_word(info, 'stairs')
            if 'steps' in info:  # Stairs
                steps_num = get_int_before_word(info, 'steps')
                info_dict['max_lifts'] = steps_num if info_dict['lifts_num'] == '' else steps_num * info_dict['lifts_num']
            if 'attempts' in info:
                info_dict['attempts'] = get_int_before_word(info, 'attempts')
            if 'kg' in info:
                info_dict = handle_weight_info(info, info_dict)
            if 'deadlift' in event_name_lower and ('deadlift' in info_lower or 'bar' in info_lower or 'tire' in info_lower):
                info_dict['deadlift'] = info
            if 'x' in info:  # Stones, medleys etc.
                implements_num, lifts_num = get_implement_and_lifts_num(info_lower)
                if info_dict['implements_num'] == '' and implements_num != '':
                    info_dict['implements_num'] = implements_num
                info_dict['max_lifts'] = lifts_num if info_dict['max_lifts'] == '' else lifts_num * info_dict['max_lifts']
        transformed_event_info.append(info_dict)

    # Add missing events, that are not in event_info csv
    all_event_names = list(event_results_df.columns)
    events_with_info = list(event_info_df['Event'])
    missing_events = [(event, i) for i, event in enumerate(all_event_names) if event not in events_with_info]
    for event_name, i in missing_events:
        info_dict = info_schema.copy()
        info_dict['event_name'] = event_name
        transformed_event_info.insert(i, info_dict)

    transformed_event_info_df = pd.DataFrame(transformed_event_info)
    return transformed_event_info, transformed_event_info_df


def transform_results(event_results_df, event_points_df, transformed_event_info, competitors):
    """Get the main and second measurement unit. Main measurement unit is assumed to be the same as the best event result.
    Second measurement unit is the most common one, that's not main unit. """

    cols = ['competitor'] + list(event_results_df.columns)
    unsorted_result_points_df = event_points_df.copy()

    final_result_values_df = pd.DataFrame(columns=cols)
    final_result_units_df = pd.DataFrame(columns=cols)
    final_result_points_df = pd.DataFrame(columns=cols)
    final_result_values_df['competitor'] = competitors
    final_result_units_df['competitor'] = competitors
    final_result_points_df['competitor'] = competitors

    for i, event_name in enumerate(event_results_df):
        event_name = event_name.strip()
        results = event_results_df[event_name].values
        points = event_points_df[event_name].values

        # Sort results by points
        zipped_results = sorted(list(zip(points, results)), reverse=True, key=lambda x: x[0])
        sorted_result_points, sorted_results = [list(t) for t in zip(*zipped_results)]
        # print(event_name, 'sorted_results:', sorted_results)
        cleaned_sorted_results = [preclean_result(txt) for txt in sorted_results]
        cleaned_unsorted_results = [preclean_result(txt) for txt in results]

        print(cleaned_sorted_results)
        # Main measurement unit
        sorted_units = extract_units_from_results(cleaned_sorted_results)
        unsorted_units = extract_units_from_results(cleaned_unsorted_results)
        transformed_event_info[i]['main_measurement_unit'] = sorted_units[0]

        # Second measurement unit
        filtered_units = [unit for unit in sorted_units if unit != sorted_units[0]]
        if non_main_units_count := Counter(filtered_units).most_common():
            transformed_event_info[i]['second_measurement_unit'] = non_main_units_count[0][0]

        final_results, unsorted_units = extract_results(cleaned_unsorted_results, unsorted_units, transformed_event_info[i])

        # Update event results
        final_result_points_df[event_name] = points
        final_result_units_df[event_name] = unsorted_units
        final_result_values_df[event_name] = final_results

        # print(unsorted_result_points_df[event_name].tolist())
        # print('units:', sorted_units, filtered_units, unsorted_units, sorted_units[0])

    return final_result_points_df, final_result_units_df, final_result_values_df, pd.DataFrame(transformed_event_info)

def preclean_result(result):
    """Clean result before result unit is extrapolated from the raw text"""
    cleaned_result = remove_text_inside_braces(result).replace('~', '').strip().lower()

    if is_nan(result):
        return cleaned_result

    if '-' in result: # 2003 Arnold's Strength Summit - Apollon Wheels had results in format of clean-continentals-presses, we need presses + reps str
        cleaned_result = f'{cleaned_result.split('-')[-1]} reps'
    if 'All' in result: # 2005 Arnold's Strongest Man - Hammer lift had weird result formatting - Rd 2, All (20.45) -> 20.45s when All
        cleaned_result = f'{get_text_inside_braces(result)} s'
    if 'Hole' in result: # 2005 Arnold's Strongest Man - Hammer lift had weird result formatting - Rd 2, All (20.45) -> 20.45s when All
        cleaned_result = f'{get_text_after_word(cleaned_result, "hole")} points'

    return cleaned_result

def extract_units_from_results(results: list) -> list:
    """Extracts units from a list of results.

    Args:
        results (list): A list of strings representing the event results.
    Returns:
        list: A list of units extracted from the results.
    """

    all_units = ['m', 's', 'kg', 'point', 'points', 'rep', 'reps', 'stair', 'stairs', 'implement', 'implements', 'stones', 'bags']
    unit_map = {
        "reps": ['rep', 'reps'],
        "implements": ['implement', 'implements'],
        "points": ['point', 'points'],
        "bags": ['bag', 'bags'],
        "stones": ['stone', 'stones'],
        "stairs": ['stair', 'stairs'],
        "degrees": ['°'],
        "kg": ['kg'],
        "m": ['m'],
        "s": ['s']
    }
    flattened_unit_map = flatten_dict(unit_map)
    units_raw = [filter_str(txt, all_units) for txt in results]
    units = [flattened_unit_map.get(unit, '') for unit in units_raw]

    return units

def flatten_dict(map: dict) -> dict:
    """Flattens a dictionary by swapping keys and values.

    Args:
        map (dict): A dictionary to be flattened.
    Returns:
        dict: A new dictionary with keys and values swapped.
    Examples:
        >>> flatten_dict({'a': [1, 2], 'b': [3, 4]})
        {1: 'a', 2: 'a', 3: 'b', 4: 'b'}
    """

    flattened_dict = {}
    for key, values in map.items():
        for value in values:
            flattened_dict[value] = key
    return flattened_dict

def extract_results(results, units, preprocessed_event_info):
    """Extract result values from the scraped tables.
        Formats:
        - <value> <unit>, f.e: 15.3 m, 12.1 s
        - <value> reps in <unit>, f.e: 5 reps in 25.3 s"""
    if ' in ' not in results[0]:
        return clean_results(results), units

    result_lifts, result_measurements = zip(*(result.split(' in ') for result in results if ' in ' in result))
    result_measurements = [to_float(measurement.split(' ')[0]) for measurement in result_measurements]
    result_lifts = [int(lift) for lift in result_lifts]
    num_of_results_with_in = len(result_lifts)

    if preprocessed_event_info['max_lifts'] == '':
        preprocessed_event_info['max_lifts'] = result_lifts[0]
    else:
        preprocessed_event_info['max_lifts'] = max(result_lifts[0], preprocessed_event_info['max_lifts'])

    # Finished event results
    finished_result_lifts = [lift for lift in result_lifts if lift == preprocessed_event_info['max_lifts']]
    finished_result_measurements = result_measurements[:len(finished_result_lifts)]

    # Unfinished event results
    unfinished_results = result_lifts[len(finished_result_lifts):] + results[num_of_results_with_in:]

    # Update units for unfinished events
    if unfinished_results:
        preprocessed_event_info['second_measurement_unit'] = 'reps' if preprocessed_event_info['second_measurement_unit'] == '' else preprocessed_event_info[
            'second_measurement_unit']
        # If main measurement unit is empty replace it with the second measurement unit
        if is_nan(preprocessed_event_info['main_measurement_unit']):
            preprocessed_event_info['main_measurement_unit'] = preprocessed_event_info['second_measurement_unit']
            preprocessed_event_info['second_measurement_unit'] = math.nan

        num_of_finished_results = len(finished_result_lifts)
        units = units[:num_of_finished_results] + [preprocessed_event_info['second_measurement_unit']] * (len(units) - num_of_finished_results)

    final_results = clean_results(finished_result_measurements + unfinished_results)
    # print(finished_result_lifts, unfinished_results, final_results, units)

    return final_results, units


def clean_result(result):
    numbers = get_floats(str(result))
    return numbers[0] if numbers else 0


def clean_results(results):
    return [clean_result(result) for result in results]


def split_results(results_df):
    """Split results df into event_results and event_points"""

    competitor_cols = ['#', 'Competitor', 'Country']
    event_point_cols = [col for col in results_df.columns if 'pts' in col.lower()]
    non_event_result_cols = competitor_cols + event_point_cols
    event_result_cols = [x for x in list(results_df.columns) if x not in non_event_result_cols]

    event_results_df = results_df[event_result_cols]
    event_points_df = results_df[event_point_cols]

    cols = event_points_df.columns.to_list()
    cols_reordered = cols[1:] + [cols[0]]
    event_points_df = event_points_df[cols_reordered]
    event_points_df.columns = event_result_cols + [event_points_df.columns[-1]]

    return event_results_df, event_points_df, results_df['Competitor']


def get_implement_and_lifts_num(event_info):
    """Calculate implement number and lift number for medleys and stones, based on implements listed (if its not there already) and '2x' or '3x' in the info. Results look like '5 in 18.24'."""

    if ',' in event_info:
        implements = [s for s in event_info.split(',') if 'kg' in s]
    elif '+' in event_info:
        implements = [s for s in event_info.split('+') if 'kg' in s]
    elif 'x' in event_info:
        implements = [event_info]
    else:
        return '', ''

    implements_num = len(implements)
    lifts_num = 0
    for words in implements:
        # Implement lifted multiple times (ex. 3x, 2x..)
        word = [word.replace('x', '') for word in words.split(' ') if 'x' in word and is_float(word.replace('x', ''))]
        if word:
            lifts_num += to_float(word[0])
        else:
            lifts_num += 1

    return implements_num, lifts_num


def get_lift_num(info):
    """Look for 'x' in the info, if there is a number in front of it then it specifies number of lifts."""
    return len([word for word in info if 'x' in info and is_float(info.replace('x', ''))])


def get_implements(split_str, info):
    return [x.split('kg')[1].strip() for x in info.split(split_str) if 'kg' in x]


def handle_weight_info(info, info_dict):
    weights = get_weights(info)
    # print(info, info_dict, weights)
    info_dict['min_weight'] = min(weights)
    info_dict['max_weight'] = max(weights)
    info_dict['weights'] = weights

    return info_dict

def get_text_after_word(s, word):
    """First split by the word and then get the first string before the word that we split on. [-2] is there because split() adds an empty string at the end"""
    return s.split(word)[1].split(' ')[-1].strip()

def get_float_before_word(s, word):
    """First split by the word and then get the first string before the word that we split on. [-2] is there because split() adds an empty string at the end"""
    return float(s.split(word)[0].split(' ')[-2].strip())


def get_int_before_word(s, word):
    """First split by the word and then get the first word before the word that we split on. [-2] is there because split() adds an empty string at the end"""
    num = s.split(word)[0].split(' ')[-2].strip()
    return to_int(num, default=math.nan)



def read_data(events_path, results_path):
    events_df = pd.read_csv(events_path, sep=',')
    results_df = pd.read_csv(results_path, sep=',')

    events_df = prepapre_df_for_merging(events_df, events_path, results_path)
    results_df = prepapre_df_for_merging(results_df, events_path, results_path)
    events_df['Event'] = events_df['Event'].str.strip()

    return events_df, results_df


def prepapre_df_for_merging(df, events_path, results_path):
    """In ordered to merge data, whether it's points, units, values or event info"""

    old_cols = list(df.columns)
    year = get_ints(results_path)[0]
    comp_name = remove_numbers(os.path.splitext(os.path.basename(events_path))[0])

    df['year'] = year
    df['competition_name'] = comp_name
    new_cols_order = ['year', 'competition_name'] + old_cols
    df = df[new_cols_order]

    return df


def merge_data(src, dst):
    result_points_dfs = []
    result_units_dfs = []
    result_values_dfs = []
    event_info_dfs = []
    raw_results_dfs = []

    # dir_path = os.path.join(dst, root.lstrip(src).replace('events', '').strip('\\'))
    create_dir(dst)

    default_cols = ['Year', 'Comp name', '#', 'Competitor', 'Country']

    for root, dirs, files in os.walk(src):
        if 'results' in root:
            continue

        for name in files:
            events_path = os.path.join(root, name)
            results_path = events_path.replace('events', 'results')
            print(results_path)

            transformed_result_points_df, transformed_result_units_df, transformed_result_values_df, transformed_event_info_df, transformed_raw_results_df = transform_csv(results_path, events_path)

            result_points_dfs.append(transformed_result_points_df)
            result_units_dfs.append(transformed_result_units_df)
            result_values_dfs.append(transformed_result_values_df)
            event_info_dfs.append(transformed_event_info_df)
            raw_results_dfs.append(transformed_raw_results_df)

    # Merge dfs
    result_points_merged_df = pd.concat(result_points_dfs, axis=0)
    result_units_merged_df = pd.concat(result_units_dfs, axis=0)
    result_values_merged_df = pd.concat(result_values_dfs, axis=0)
    event_info_merged_df = pd.concat(event_info_dfs, axis=0)
    raw_results_merged_df = pd.concat(raw_results_dfs, axis=0)

    # Save dfs
    result_points_merged_df.to_csv(os.path.join(dst, 'result_points.csv'), sep=';', index=False)
    result_units_merged_df.to_csv(os.path.join(dst, 'result_units.csv'), sep=';', index=False)
    result_values_merged_df.to_csv(os.path.join(dst, 'result_values.csv'), sep=';', index=False)
    event_info_merged_df.to_csv(os.path.join(dst, 'event_info.csv'), sep=';', index=False)
    raw_results_merged_df.to_csv(os.path.join(dst, 'raw_results.csv'), sep=';', index=False)



def filter_str(s, word_list, exclude=False):
    """If exclude is True remove words in string that are in the word list
        If exclude is False then only keep the words in string that are in the word list"""
    if exclude:
        return ' '.join([word for word in s.split(' ') if word.lower() not in word_list]).strip()
    else:
        return ' '.join([word for word in s.split(' ') if word.lower() in word_list]).strip()


def get_weights(txt):
    txt_cleaned = txt.replace(',', '').replace('+', '').replace('-', ' ')
    words = txt_cleaned.split(' ')
    weight_key_words = ['kg', 'to']
    return [float(words[i - 1]) for i in range(1, len(words)) if words[i] in weight_key_words and is_float(words[i - 1])]


def get_floats(txt):
    return [float(x) for x in txt.split(' ') if is_float(x)]


def get_ints(txt):
    return [int(x) for x in re.findall("\d+", txt)]


def remove_numbers(txt):
    return re.sub(r'[0-9]+', '', txt).strip()


def remove_punctuation(txt):
    if is_nan(txt):
        return ''
    return re.sub(r'[^\w\s]', '', txt)


def remove_braces(txt):
    if is_nan(txt):
        return ''
    return re.sub("[(\[].*?[)\]]", "", txt)


def remove_text_inside_braces(txt):
    if is_nan(txt):
        return ''
    return re.sub("[\(\[].*?[\)\]]", "", txt)

def get_text_inside_braces(txt):
    return txt[txt.find("(") + 1:txt.find(")")]

def get_float_inside_braces(txt):
    return to_float(txt[txt.find("(") + 1:txt.find(")")])

def is_nan(x):
    return x != x


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def is_int(num):
    try:
        int(num)
        return True
    except ValueError:
        return False


def to_int(s, default=''):
    """Convert string to int safely."""
    return float(s) if is_int(s) else default


def to_float(s, default=''):
    """Convert string to float safely."""

    return float(s) if is_float(s) else default


s= "HafthÃ³r JÃºlÃ­us 'The Mountain' BjÃ¶rnsson"
s2="Ã�rvai".replace("\u00C3\uFFFD", "A").replace("\u00E1\uFFFD", 'a').replace("\uFFFD", '?')
s4="I. Árvai"
s3="M. Ver MagnÃºsson"
s5="Ã�rvai"

print(s5)
# print(s3.encode('windows-1252').decode('utf-8'))
print(fix_encoding(s5))
print(fix_encoding(s5).replace("�", "Á"))


# parse_competition("2004 Arnold's Strongest Man", ARNOLD_CLASSIC_URL, 'arnold_classic', is_wsm=False, force_update=True)
# parse_competition("1995 World's Strongest Man", WSM_URL, 'world_strongest_man', is_wsm=True, force_update=True)

# parse_all_competitions(ARNOLD_CLASSIC_URL, 'arnold_classic', is_wsm=False, force_update=True)
# parse_all_competitions(WSM_URL, 'world_strongest_man', is_wsm=True, force_update=True)
# parse_all_competitions(ROGUE_INVITATIONAL_URL, 'rogue_invitational', is_wsm=False, force_update=True)
# parse_all_competitions(GIANTS_URL, 'giants', is_wsm=False, force_update=True)
# parse_all_competitions(WUS_URL, 'wus', is_wsm=False, force_update=True)
# parse_all_competitions(SHAW_CLASSIC_URL, 'shaw_classic', is_wsm=False, force_update=True)
# parse_all_competitions(FORCA_BRUTA_URL, 'forca_bruta', is_wsm=False, force_update=True)
# parse_all_competitions(ULTIMATE_STRONGMAN_URL, 'ultimate_strongman', is_wsm=False, force_update=True)


# events_path = '../data/data_raw/rogue/events/2021 Rogue Invitational.csv'
# events_path = '../data/data_raw/world_strongest_man/events/finals/2021 WSM Final.csv'
# events_path = '../data/data_raw/world_strongest_man/events/groups/2017 WSM Final - group 1.csv'

# events_path = '../data/data_raw/arnold_classic/events/2020 Arnold Strongman Classic.csv'

# events_path = "../data/data_raw/arnold_classifiers/events/2018 Arnold South America.csv"
# results_path = events_path.replace('events', 'results')
# transform_csv(results_path, events_path)
# #
merge_data(DATA_RAW_DIR_PATH, DATA_TRANSFORMED_DIR_PATH)
#

# Get 2 units from 1 result
