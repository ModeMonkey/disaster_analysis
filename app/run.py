import joblib
import json
import pandas as pd
import plotly
import re

from flask import Flask
from flask import render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from plotly.graph_objs import Bar
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine

app = Flask(__name__)

# Functions used in the pipeline need ot be present
# so that the pickled model can reference them for
# new data

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('merged_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_columns = df.columns[4:]
    category_sums = []
    for category in category_columns:
        category_sum = sum(df[category] == 1)
        category_sums.append(category_sum)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, {
            'data': [
                Bar(
                    x=category_columns,
                    y=category_sums
                )
            ],

            'layout': {
                'title': 'Number Positive Hits Per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
