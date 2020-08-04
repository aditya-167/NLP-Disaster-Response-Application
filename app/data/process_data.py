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
    categories = df['categories'].str.split(';',expand = True)

    # selecting the first row of the categories dataframe to extract column names
    row = categories.loc[0]

    # creating a list of category column names
    category_colnames = [category[:len(category)-2] for category in row ]

    # renaming the columns of `categories` dataframe
    categories.columns = category_colnames

    # now converting category values to just numbers 0 or 1
    for column in categories:

    	# setting each value to be the last character of the string
    	categories[column] = categories[column].str[-1]
    
    	# converting column from string to numeric
    	categories[column] = pd.to_numeric(categories[column])

    # replacing categories column in df with new category columns.
    df = df.drop('categories',axis = 1)
    
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)

    # removing duplicates
    cleaned_df = df.drop_duplicates(keep = 'first')
    #convert to binaries by dropping '2'
    df.drop(df.loc[df['related']==2].index, inplace=True)
    print(df["related"].value_counts())
    return cleaned_df


def save_data(df, database_file):
    """
    Save Data function
    
    Arguments:
        df : Clean dataframe
        database_filename : database file (.db) destination 
    return : None
    """
    engine = sq.create_engine('sqlite:///'+ database_file)
    df.to_sql("ResponseTable", engine, index=False,if_exists='replace')
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
