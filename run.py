import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk
import os
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sqlalchemy import create_engine
from .utils import tokenize

app = Flask(__name__)

class Begin_verb(BaseEstimator, TransformerMixin):
  
    def begin_verb(self, text):
        sentence = nltk.sent_tokenize(text)
        for i in sentence:
            pos_tags = nltk.pos_tag(tokenize(i))
            f_word, f_tag = pos_tags[0]
            if f_tag in ['VB', 'VBP'] or f_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tag = pd.Series(X).apply(self.begin_verb)
        return pd.DataFrame(X_tag)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.sav")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:,4:].columns
    category_boolean = (df.iloc[:,4:] != 0).sum().values
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean
                )
            ],

            'layout': {
                'title': 'Message categories',
                'yaxis': {
                    'title': "frequency"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 30
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


#def main():
#    app.run(debug=True)


#if __name__ == '__main__':
#    port = int(os.environ.get("PORT", 5000))
#    app.run(host='0.0.0.0', port=port)
