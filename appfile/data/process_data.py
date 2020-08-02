import sys
import numpy as np
import pandas as pd
import sqlalchemy as sq

def load_data(messages_file, categories_file):
    """
    Load Data Function
    
    Arguments:
        messages_file : path to messages csv file
        categories_file : path to categories csv file
    return:
        df : Loaded data as Pandas DataFrame
    """
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    Clean Data function
    
    Arguments:
        df : data Pandas DataFrame
    return:
        df : clean data Pandas DataFrame
    """
    categories = df.categories.str.split(pat=';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1) 
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_file):
    """
    Save Data function
    
    Arguments:
        df : Clean dataframe
        database_filename : database file (.db) destination 
    return : None
    """
    engine = sq.create_engine('sqlite:///'+ database_file)
    df.to_sql('df', engine, index=False)
    pass  


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the  and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
