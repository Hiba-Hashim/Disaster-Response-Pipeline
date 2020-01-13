import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function load the messages and categories datasets
    and return the mergeing value of both
    
    Args:
    (string) messages_filepath -- path of the 'messages.csv'
    (string) categories_filepath -- path of the 'categories.csv' 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
   
    return pd.merge(messages, categories, on='id')
    

def clean_data(df):
   """
   this function take a data frame and :
   - split & and clean up the categories column
   - drop duplicate rows
   - remove missing values
   
   and than return a clean dataframe with split categories
   
   Args:
   (dataframe) df -- the merged messages and categories data frame
   """

   # Split categories into separate category columns
   categories = df.categories.str.split(';', expand=True)
   row = categories[:1]
    
   # extract category names & rename the columns
   category_colnames = row.applymap(lambda s: s[:-2]).iloc[0, :].tolist() 
   categories.columns = category_colnames
    
   for column in categories:
    
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    
   # remove the old categories colum and add the new one
   df.drop('categories', axis=1, inplace=True)
   df = pd.concat([df, categories], axis=1)

   # remove duplicates
   df[df.duplicated(subset='message')].count()

   # remove missing values
   return df


def save_data(df, database_filename):
    """
    this function save the resulting dataframe to an sqlite data base
    
    Args:
    (dataframe) df -- dataframe you want to save
    (string) database_filename -- the file path to save the database in
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_category', engine, index=False, if_exists='replace')

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
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()