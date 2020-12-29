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
    '''
    input previously built df from ETL
    output defined X, Y that will be used for ML pipeline
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table ('DisasterResponse', con=engine)
#let's perform some cleaning
#all we need, is the message column, and the categories
#drop the unnecessary ones, but in case the final datasets in IDE somewhat differs in shape let's keep the columns we want instead
    X=df['message'].values
    category_names = df[['aid_centers', 'aid_related', 'buildings', 'child_alone', 'clothing','cold', 'death', 'direct_report', 'earthquake', 'electricity', 'fire','floods', 'food', 'hospitals', 'infrastructure_related','medical_help', 'medical_products',  'military','missing_people', 'money', 'offer', 'other_aid','other_infrastructure', 'other_weather', 'refugees', 'related','request', 'search_and_rescue', 'security', 'shelter', 'shops', 'storm','tools', 'transport', 'water', 'weather_related']].columns
    Y=df[category_names]
    return X,Y, category_names
    pass


#next we'll build the tokenize function
def tokenize(text):
    #remove all punctuations and irregular characters and convert everything into lower letters
    '''
    input any free text that we would like to put into our ML pipeline
    output a set of tokens converted from the text using tokenizer and lemmatizer
    '''
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
    '''
    input previously defined tokenize function
    output a ML pipeline that makes use of tiidf, Multioutputclassifier, RFclassifier, and GridSearch
    '''
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


def evaluate_model(model, X_test, Y_test, category_names):
#train our model using training data - we will use a 30%/70% testing/training split
#let's train the pipeline using the built pipeline and training data
    '''
    input previously built pipeline model, test datasets, previously built category names which are lables for performance report
    output performance report
    '''
#test
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))
    


def save_model(model, model_filepath):
#saving the model as a pickle file so that we can deploy that into the web app
    with open(model_filepath, 'wb') as pkl_file:
                  pickle.dump(model, pkl_file)
    pkl_file.close()

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #let's use a 30%/70% split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print('Building model...')
        model = pipeline_GS()
        
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

 