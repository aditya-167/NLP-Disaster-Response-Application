# Disaster Response Pipeline Project


# Disaster Response ETL, ML pipelines and Web application.

### Motivation
The purpose of the project is to build a model for an API that classifies disaster messages.
Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.  

The web app also displays visualizations of the data.

## Web application screenshots

##### Homepage

![file3](https://github.com/aditya-167/NLP-Disaster-Response-Application/blob/master/Screenshots/homepage.png)

##### message classified response

![file3](https://github.com/aditya-167/NLP-Disaster-Response-Application/blob/master/Screenshots/query.png)


### Install
This project requires Python 3.x and the following Python libraries installed:

Run : - $pip install -r requirements.txt

1. NumPy
2. Pandas
3. Matplotlib
4. Json
5. Plotly
6. Nltk
7. Flask
8. Sklearn
9. Sqlalchemy
10. Sys
11. Re
12. Pickle


### Code and data

##### Directories

1. `Notebooks` :- ipybn notebooks for ETL,NLP and model.

2. `app/data` :- contains all the csv files, ETL Pipeline scripts and.db database files from Figure8 

3. `app/models` : contains saved model classifier.pkl file as well as training model script.

-  `process_data.py`: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.

-  `train_classifier.py`: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

-  ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py  automates this notebook.

-  ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.

-  `disaster_messages.csv`, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.


### To train and run from scratch:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python app/data/process_data.py app/data/disaster_messages.csv app/data/disaster_categories.csv app/data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python app/models/train_classifier.py app/data/DisasterResponse.db app/models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:5001


##### Run app locally
In a terminal navigate to the top-level project directory udacity-disaster-response/ (that contains this README) and run commands in the following sequence:

-  python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db  
-  python train_classifier.py DisasterResponse.db classifier.pkl
-  python run.py


##### Web application on Heroku
Web Application Link - https://my-disaster-app.herokuapp.com/

### Data observations
As can be seen from test data visualization most of the classes (categories) are highly imbalanced. This affects model F1 prediction score. One using this project should take this into consideration and apply measures like synthetic data generation, model selection and parameters fine-tuning, etc.     


