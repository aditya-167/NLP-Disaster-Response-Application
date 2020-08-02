import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import fbeta_score, classification_report
from scipy.stats.mstats import gmean
import pandas as pd
import pickle
from sqlalchemy import create_engine
import re
import numpy as np
import nltk
from herokutokenizer import Tokenizer, StartingVerbExtractor


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

def load_data(database_filepath):
    """
    Load Data Function
    
    Arguments:
        database_filepath : path to SQLite db
    return:
        X : feature DataFrame
        Y : label DataFrame
        category_names -> used for app
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("ResponseTable",engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

'''
class Begin_verb(BaseEstimator, TransformerMixin):
    """
    initial beginning verb extractor class to extract verbs from 
    sentences to make a new feature. (More on this in ETL.ipynb)
    A custom class implementation from sklearn BaseEstimator for better
    results.
    """

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


'''

def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP and apply a classifier, here I have used Adaboost classifier.
    """
    model = Pipeline([
        ('features', FeatureUnion([

            ('textpipeline', Pipeline([
		('tokenizer', Tokenizer()),
                ('vectorize', CountVectorizer()),
                ('Tfidf', TfidfTransformer())
            ])),

            ('begin_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return model

def F1_score(y_true,y_pred,beta=1):
    """
    F1_score function
    
    This is a performance metric of custom F1 score model
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true : labels
        y_prod : predictions
        beta : beta value of fscore metric
    
    Output:
        f1score : customized fscore
    """
    scores = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        scores.append(score)
    f1score_numpy = np.asarray(scores)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    return  f1score

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function is to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
        model : Scikit ML Pipeline
        X_test : test features
        Y_test : test labels
        category_names : label names (multi-output)
    """
    Y_pred = model.predict(X_test)
    
    f1 = F1_score(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score  {0:.2f}%\n'.format(f1*100))


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    


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
