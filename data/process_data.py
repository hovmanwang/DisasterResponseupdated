#this .py file is essentially a summarized version of the ipynb file for ETL, which was built in Udacity's workspace
#ipynb file is available in the GitHub repo in case needed for reference


# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine



#step 1: load the raw data and produce a combined df
def load_data(messages_filepath, categories_filepath):
#we'll load the provided 2 csv files first, then merge them together
    
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
#we'll change the order of things a little. We'll just load the 2 files, then perform spliting of categories, then finally merge into df
    return messages,categories
    
    
    pass


def clean_data(messages,categories):
# create a dataframe of the 36 individual category columns
    categories_expanded = pd.DataFrame(categories['categories'].str.split(";", expand=True))
# select the first row of the categories dataframe, which will be used to rename the categories_expanded df
    row = categories_expanded.iloc[0]
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
# rename the columns of categories_expanded 
    categories_expanded.columns = category_colnames
# set each value to be the last character of the string which is integer 0 or 1, then convert to int type
    for column in categories_expanded:
        categories_expanded[column] = categories_expanded[column].astype(str).str[-1].astype(int)
#join categories_expanded back to categories using concat
    categories_updated=pd.concat([categories, categories_expanded],axis=1)
#then drop the original categories column
    categories_updated=categories_updated.drop(['categories'],axis=1)
    categories=categories_updated
#using id as the join condition, derive the df_final dataset using categories_updated dataset
    df = messages.merge(categories, how='inner', on=['id'])
#getting rid of the duplicates
    df=df.drop_duplicates(keep='first')
    return df
    pass

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)

    
  

    
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)
        
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

  