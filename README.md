# DisasterResponseupdated
updated repo for 2nd project DR
# Udacity Data Scientist 2nd Project Data Engineering


Udacity project:
project purpose is to build a flask app which makes use of news, direct and social media messages with classification, apply ETL and ML pipeline, in order to build a model that classifies any messages into a disaster category.

Building this model will enable a user to understand what is the current hot topic and response promptly, for example if there is a water shortage then he can start buying bottled water. For organizations this will allow swift response and monitoring of any disaster situation and in turn deploy support in events such as earthquakes.

input files: messages.csv and categories.csv

Following libraries were required:

sqlalchemy
sqllite3
sklearn
pandas
numpy
nltk
pickle



Project components:
The project consists of three components

1. ETL Pipeline: a procedure for Extraction Transformation and Load, making use of the 2 input files. Output of this stage is a clean dataframe that can be used for Machine Learning Pipeline


2. ML Pipeline: using the dataframe from ETL, build a Machine Learning model that makes use of Tokenization, Tf-Idf, and Gridsearch, that classifies the model into one or more of the categories


3. Flask web app: using the trained model from ML Pipeline, build a web app that is user-friendly and can help classify any messages input by the user


Here's the file structure of the project (taken from Udacity Project):

app


|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- DisasterResponse.db # database to save clean data to
|- ETL Pipeline Preparation (2) - the ipynb file which is directly downloaded from Udacity workspace, with results

models

|- ML Pipeline Preparation.py - directly downloaded from Udacity workspace, with results
|- ML Pipeline Preparation (2).ipynb - directly downloaded from Udacity workspace, with results
|- classifier.pkl # saved model


|- README.md


|-Instructions:
|-Run the following commands in the project's root directory to set up your database and model.

|-To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
|-To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
|-Run the following command in the app's directory to run your web app. python run.py

|-Go to http://0.0.0.0:3001/, my personal link is https://view6914b2f4-3001.udacity-student-workspaces.com/



Acknowledgements

Data provided by FigureEight
