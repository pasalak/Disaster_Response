import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load_data
    load data from messages and categories csv files into single pandas dataframe
    
    INPUT
        messages_filepath : data/disaster_messages.csv
        categories_filepath: data/disaster_categories.csv
   
    OUTPUT
     df retruns consolidated (merging) categories and messages
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df
   


def clean_data(df):
    """
    Clean data includes transform categories part
    INPUT
        df : Merged DataFrame
    
    OUTPUT
         df: Returns Cleaned DataFrame
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda i: i[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] =  categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    df.drop(columns = ['categories'], inplace=True)
    df = df.join(categories)
    # Deleting the value 2 as per reviews comments.
    df.drop(df[df['related'] == 2].index, inplace = True)
    df = df.drop_duplicates()
    return df
   


def save_data(df, database_filename):
    """
    Stores a df in a SQLite database
    input:
        df: a pandas Data Frame
        Table name : DisasterResponse
        database_filename: DisasterResponse.db
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine , if_exists='replace', index = False)
    


def main():
    """
    Main function will initiate data load,clean and save data methods.
    input:
        df: a pandas Data Frame
        Table name : DisasterResponse
        database_filename: DisasterResponse.db
    
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
