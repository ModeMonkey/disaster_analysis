"""
Script is used to train a model to predict the cateogries 
of messages.
Script performs the following:
    1) Loads the data from an SQL database.
    2) Splits the data into train and testing sets.
    3) Trains the model on multiple parameters using GridSearchCV.
    4) Prints the performance of the model against each category.
    5) Saves the model to a pickle file.
"""

import sys
import re
import operator

import pandas as pd
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine

def load_data(database_filepath="./data/data.db"):
    """
    Load data from an sql store into a dataframe
    In:
        database_filepath = filepath to sql store on disk
    Out:
        X, Y dataframes and Y_columns
        X dataframe has the message and genre columns
        Y dataframe has the categories and associated values
        that are to be predicted
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    query = "SELECT * FROM merged_data"
    df = pd.read_sql(query, engine)
    X_columns = ['id', 'message', 'original', 'genre']
    #X = df[["message", "genre"]] # Deprecated.  Used when using build_model_2
    X = df["message"]
    Y_columns = list(set(df.columns) - set(X_columns))
    Y = df[Y_columns]
    return X, Y, Y_columns

def tokenize(text):
    """
    Strip all non-alpha characters from string, lower, strip.
    Then tokenize, lemmatize, and remove stopwords from string.
    In:
        text = string with multiple words and sentences.
    Out:
        tokens that have been generated from text
    """
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("  +", " ", text)
    text = text.lower()
    text = text.strip(" ")
    all_tokens = [token for sent in sent_tokenize(text) for token in word_tokenize(sent)]
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(tok) for tok in all_tokens]
    stopwords_set = set(stopwords.words("english"))
    out_tokens = [tok for tok in lemma_tokens if tok not in stopwords_set]
    return out_tokens

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Select columns to be used in subsequent methods
    in:
        key = columns to keep in a list form
    out:
        transformer:
            Dataframe with only the selected columns
            Series if only only column is selected
    """
#
    def __init__(self, key):
        self.key = key
#
    def fit(self, x, y=None):
        return self
#
    def transform(self, X):
        return X[self.key]

class PatternMatcher(BaseEstimator, TransformerMixin):
    """
    Select columns to be used in subsequent methods
    in:
        pattern = pattern to match against
        Example: PatternMatcher(pattern="\!")
    out:
        transformer:
            Series of True/False values stating if
            pattern was found
    """
    def __init__(self, pattern):
        self.pattern = pattern
#
    def fit(self, x, y=None):
        return self
#
    def transform(self, X):
        X_tagged = pd.Series(X).str.contains(self.pattern)
        return pd.DataFrame(X_tagged)

class GenreClassifier(BaseEstimator, TransformerMixin):
    """
    Deprecated.  Preserved for prosperity. 
    Relys on two columns of data from the messages dataset - message and Genre.
    One hot encoding the values of the Genre column does not improve the model's
    ability to predict associated categories of a message.
    """
#
    def fit(self, x, y=None):
        return self
#
    def transform(self, X):
        direct_mask = pd.Series(X).str.contains("direct")
        news_mask = pd.Series(X).str.contains("news")
        social_mask = pd.Series(X).str.contains("social")
        out_df = pd.DataFrame()
        out_df["direct"] = direct_mask
        out_df["news"] = news_mask
        out_df["social"] = social_mask
        return out_df

def build_model_2():
    """
    Deprecated.  Kept for prosperity.
    Does not perform better than build_model but requires additional
    processing.
    Relys on two columns of data from the messages dataset - message and Genre.
    One hot encoding the values of the Genre column does not improve the model's
    ability to predict associated categories of a message.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ("text_pipeline_column", ColumnSelector(key="message")),
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('exclamation_pipeline', Pipeline([
                ("exclamation_pipeline_column", ColumnSelector("message")),
                ('exclamation_present', PatternMatcher(pattern=r"\!")),
            ])),
            ('question_pipeline', Pipeline([
                ("question_pipeline_column", ColumnSelector("message")),
                ('question_mark_present', PatternMatcher(pattern=r"\?")),
            ])),
            ('genre_pipeline', Pipeline([
                ("genre_pipeline_column", ColumnSelector(key="genre")),
                ('genre_pipeline_columns', GenreClassifier()),
            ])),
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [2, 3, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1)
    return cv


def build_model():
    """
    Builds model for the pipeline.  
    Tokenizes the messages in the message column.
    Also records if a quesiton or exclamation mark
    was present in a message.
    Allows for multiple categories to be predicted
    by using the  MultiOutputClassifier method. 
    Predicts a message's categories using the Random
    Forest Classifier method.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('exclamation_present', PatternMatcher(pattern=r"\!")),
            ('question_mark_present', PatternMatcher(pattern=r"\?")),
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2), (1,3)),
        'features__text_pipeline__tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [2, 3, 4, 10]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the results of the model in order of most effective
    prediction to least effective. 
    In:
        model = model after fitting
        X_test = X test data
        Y_test = Y test data
        category_names = Y data data column names
    """
    Y_pred = model.predict(X_test)
    results_map = {}
    for i in range(len(category_names)):
        category_name = category_names[i].ljust(30, ".")
        score = accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])
       
        results_map[category_name] = score
    sort_results_map = sorted(results_map.items(), key=operator.itemgetter(1), reverse=True)
    print("column.......................Score")
    for result in sort_results_map:
         print(f"{result[0]}{result[1]}")

def save_model(model, model_filepath):
    """
    Saves the model to a pickle file.
    in:
        model = model after training
        model_filepath = location to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    print(f"file saved to: {model_filepath}")

def main():
    """
    Launches script to load the data, seperate the data into
    training and test data, builds, trains, and evaluates the
    model, and saves the data to a pickle file. 
    Launch:
        python train_classifier.py path/to/source_database.db /
                                   path/to/destination_model_file.pickle
    """
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