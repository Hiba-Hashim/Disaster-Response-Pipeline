import sys
import pandas as pd
import numpy as np
import string
import re
import pickle
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report


def load_data(database_filepath):
    """
    This function load the dataset from the database
    and return the features X and target Y variables and coulumes names of the
    target 
    
    Args:
    (string) database_filepath -- path of the database
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("messages_category",engine)

    X = df['message'].values
    Y = df.iloc[:,4:].values
    
    columns_names = df.iloc[:,4:].columns
    
    return X, Y, columns_names


def tokenize(text):
    """
    This function is process your text data by removing stop words and lemmatizing the text
    and return a clean and tokeniz text
    
    Args:
    (string) text -- a piece of text
    """
   # normalize case and remove punctuation
    txet = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    
    # tokenize text
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for t in tokens:
        clean_t = lemmatizer.lemmatize(t).lower().strip()
        clean_tokens.append(clean_t)
        
    return clean_tokens


def build_model():
    """
    This function build the machine learning pipeline
    and return a machine learning model pipeline
    
    Args:
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__max_depth':[10,15],
                 'clf__estimator__min_samples_split': [2, 3]}

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, columns_names):
    """
    This function print the classification report
    
    Args:
    () model -- machine learning model
    (dataframe) X_test -- containing independent Features from the test set
    (dataframe) Y_test -- containg dependent Features from the test set
    (list) columns_names -- names of Dependent Features
    """
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(columns_names):
        print(col, classification_report(Y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    """
    This function Saves the model into a pickle file
    
    Args:
    () model -- trained machine learning model
    (string) model_filepath -- Name of the pickle file to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()