# DisasterResponseupdated
updated repo for 2nd project DR
# Udacity Data Scientist 2nd Project Data Engineering


Udacity project:
project purpose is to build a flask app which makes use of news, direct and social media messages with classification, apply ETL and ML pipeline, in order to build a model that classifies any messages into a disaster category

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

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- DisasterResponse.db # database to save clean data to

models

|- train_classifier.py

|- classifier.pkl # saved model

|- README.md

Acknowledgements

Data provided by FigureEight
