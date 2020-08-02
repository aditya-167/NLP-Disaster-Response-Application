import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(sentence.split())
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        def tokenize(text):
            """ 
	    Tokenize Function. 
	  
	    Cleaning The Data And Tokenizing Text. 
	  
	    Parameters: 
	    text (str): Text For Cleaning And Tokenizing (English).
	    
	    Returns: 
	    clean_tokens (List): Tokenized Text, Clean For ML Modeling
	    """
            # Define url pattern
            url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

            # Detect and replace urls
            detected_urls = re.findall(url_re, text)
            for url in detected_urls:
                text = text.replace(url, "urlplaceholder")

            # tokenize sentences
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()

            # save cleaned tokens
            clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

            # remove stopwords
            STOPWORDS = list(set(stopwords.words('english')))
            clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]

            return ' '.join(clean_tokens)

        return pd.Series(X).apply(tokenize).values
