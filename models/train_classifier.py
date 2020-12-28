#this .py file is essentially a summarized version of the ipynb file for ML Pipeline, which was built in Udacity's workspace
#ipynb file is available in the GitHub repo in case needed for reference - all model classification reports and other results are available there


# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import sqlite3
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle



def load_data(database_filepath):
#load the previously built data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table ('DisasterResponse', con=engine)
#let's perform some cleaning
#all we need, is the message column, and the categories
#drop the unnecessary ones, but in case the final datasets in IDE somewhat differs in shape let's keep the columns we want instead
    X=df[['message']]
    Y=df[['message',  'related', 'request','offer', 'aid_related', 'medical_help', 'medical_products','search_and_rescue', 'security', 'military', 'child_alone', 'water','food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees','death', 'other_aid', 'infrastructure_related', 'transport','buildings', 'electricity', 'tools', 'hospitals', 'shops','aid_centers', 'other_infrastructure', 'weather_related', 'floods','storm', 'fire', 'earthquake', 'cold', 'other_weather','direct_report']]
    return X,Y
    pass


#next we'll build the tokenize function
def tokenize(text):
    #remove all punctuations and irregular characters and convert everything into lower letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass


def pipeline_GS():
    pipeline_GS = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(estimator= RandomForestClassifier())) #we'll use RF under Multioutputclassifier given it's 36 cats
                    ])

    parameters = {
                    'vect__min_df':[1,10,50],
                    'tfidf__smooth_idf': [True, False],
                    }
    #apply Gridsearch here
    model  = GridSearchCV(pipeline_GS, param_grid=parameters, cv=3) 
    return model 
    pass


def evaluate_model(model, X_test, Y_test):
#train our model using training data - we will use a 30%/70% testing/training split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
#i needed to have this package in order to produce the classification report
    from sklearn.utils.multiclass import type_of_target
    type_of_target(y_test)
    type_of_target(y_pred)
    
    target_names=np.unique(y_pred)
    model=pipeline_GS()
    print(classification_report(np.hstack(y_test),np.hstack(y_pred)))
    pass




def save_model(model, model_filepath):
#saving the model as a pickle file so that we can deploy that into the web app
    with open(model_filepath, 'wb') as pkl_file:
                  pickle.dump(model, pkl_file)
    pkl_file.close()

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        #let's use a 30%/70% split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = pipeline_GS()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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