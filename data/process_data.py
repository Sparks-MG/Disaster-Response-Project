#load dependencies
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from two csv files (disaster_categories.csv and disaster_messages.csv) and merges the data.
    input: string representing path to disaster_categories.csv, string representing path to disaster_messages.csv,
    output: returns dataframe with the combined content of both provided csv files.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner',on=['id'])
    return df


def clean_data(df):
    """
    Cleans the merged dataframe from load_data by reforming and reforming columns.
    input: dataframe (merged raw output from the load_data method)
    output: dataframe (cleaned)
    """

    #Splits the different message categories in different columns of a new dataframe
    categories = df['categories'].str.split(";",expand = True)

    #reformats the colum names via list comprehension
    row = categories.iloc[0,:]
    category_colnames = [category[:-2] for category in row]
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])

    #Remove old categories, merge and clean the combined dataframe
    df= df.drop(['categories'], axis=1)
    df = pd.concat([df,categories], axis=1)

    #Optionally: Check for duplicates: pd.concat(g for _, g in df.groupby("id") if len(g) > 1)

    #drop duplicates
    df = df.drop_duplicates(keep='first')

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned data(frame) to an SQLite database
    input: cleaned dataframe, filename of the SQLite database as string
    output: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
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
        print('\nPlease provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()