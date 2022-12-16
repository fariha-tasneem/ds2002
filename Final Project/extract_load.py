import csv
from csv import writer
from csv import reader
import os
import pymongo
import pprint
import pandas as pd

# PART 1 - extract data sources into one Pandas -> only extracting specific columns

# Shows data
shows_df = movies_df = pd.read_csv(r'Best Shows Netflix.csv')
final_shows_df = shows_df[['TITLE', 'RELEASE_YEAR', 'SCORE', 'NUMBER_OF_SEASONS', 'DURATION']]

# Movies data
movies_df = pd.read_csv(r'Best Movies Netflix.csv')
final_movies_df = movies_df[['TITLE', 'RELEASE_YEAR', 'SCORE', 'MAIN_GENRE']]

# Raw titles data
raw_df = pd.read_csv(r'raw_titles.csv')
final_raw_df = raw_df[['title', 'type', 'release_year', 'age_certification']]

print(final_raw_df)

# PART 2 -  Load data into a MongoDB

host_name = "localhost"
port = "27017"
conn_str = {
    "local" : f"mongodb://{host_name}:{port}/"
}

client = pymongo.MongoClient(conn_str["local"])

db_name = "netflix_data_final"
db = client[db_name]

shows = db['kaggle_shows_data']
movies = db['kaggle_movies_data']
raw = db['kaggle_raw_data']

final_shows_df.reset_index(inplace=True)
data_dict1 = final_shows_df.to_dict("records")
shows.insert_many(data_dict1)

final_movies_df.reset_index(inplace=True)
data_dict2 = final_movies_df.to_dict("records")
movies.insert_many(data_dict2)

final_raw_df.reset_index(inplace=True)
data_dict3 = final_raw_df.to_dict("records")
raw.insert_many(data_dict3)

