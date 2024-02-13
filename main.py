# Scraping strongman data from: https://strongmanarchives.com/
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', 200)

if __name__ == '__main__':
    path = 'data/data_raw_merged/preprocessed_event_info.csv'
    df = pd.read_csv(path, sep=';')
    print(df.tail(200))

