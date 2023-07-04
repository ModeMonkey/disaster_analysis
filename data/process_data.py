"""
Script is used to load, format, and save messages and associated categories.
Script performs the following:
    1) Loads messages and associated categories from two csv's into one 
       dataframe.  Merges the two csv's along the message and categories'
       id.
    2) Formats the category column names and turns the values in each
       category into a binary format.
    2) Cleans the messages by removing duplicates.
    3) Saves the data into an SQL database. 
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads two datasets - the messages and categories datasets.
    Cleans the categories dataset by turning values into binary
    and setting the column names.
    Merges the two datasets along on the "id" column.
    In:
        messages_filepath   = str filepath to messages dataset
        categories_filepath = str filepath to categories dataset
    Out:
        pandas dataframe
    """
    df_mes = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    df_cat_values = df_cat["categories"].str.split(";", expand=True)
    df_cat_columns = [column.rsplit("-")[0] for column in df_cat_values.loc[0]]
    cat_column_map = {}
    for x in range(len(df_cat_columns)):
        cat_column_map[x] = df_cat_columns[x]
    df_cat_values = df_cat_values.rename(columns=cat_column_map)
    for column in df_cat_values.columns:
        df_cat_values[column] = df_cat_values[column].str.rsplit("-").str[1]
    df_cat = pd.concat([df_cat["id"],
                        df_cat_values],
                        axis=1)
    df_out = pd.merge(df_mes, df_cat, on='id', how='inner')
    return df_out

def clean_data(df):
    """
    removes duplicates from a dataframe
    In:
        df = dataframe
    Out:
        dataframe
    """
    columns = list(df.columns)
    columns.remove("id")
    duplicates = df[columns].duplicated(keep="first")
    df_out = df[~duplicates]
    duplicates = df_out[columns].duplicated(keep="first")
    assert sum(duplicates) == 0, "Duplicates were not removed properly"
    return df_out

def save_data(df, database_filepath):
    """
    Saves dataframe to an sql store.
    In:
        df = dataframe
        database_filepath = location on disk create and save sql store
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('merged_data', engine, index=False) 

def main():
    """
    Launches script to load the data, organize it, clean it,
    then save it to an SQL database. 
    Launch:
        python process_data.py path/to/disaster_messages.csv /
                               path/to/disaster_categories.csv /
                               path/to/receiving_database.db
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()