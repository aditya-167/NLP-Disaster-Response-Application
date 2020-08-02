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


