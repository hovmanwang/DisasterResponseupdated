#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[94]:


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
engine = create_engine('sqlite:///' + 'HWDEdf', echo=False)
import sqlite3
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


# In[70]:


# load data from database
df=pd.read_sql_table('HWDEdf', con=engine)
df.head()
#yes!


# In[71]:


df.columns


# In[58]:


#check nulls
df.isnull().sum()
#good, same as what we expect from ETL


# In[40]:


#let's perform some cleaning
#all we need, is the message column, and the categories
#drop the unnecessary ones, but in case the final datasets in IDE somewhat differs in shape let's keep the columns we want instead
df=df[['message',  'related', 'request','offer', 'aid_related', 'medical_help', 'medical_products','search_and_rescue', 'security', 'military', 'child_alone', 'water','food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees','death', 'other_aid', 'infrastructure_related', 'transport','buildings', 'electricity', 'tools', 'hospitals', 'shops','aid_centers', 'other_infrastructure', 'weather_related', 'floods','storm', 'fire', 'earthquake', 'cold', 'other_weather','direct_report']]


# In[5]:


df.head()
#good, exactly what we want


# ### 2. Write a tokenization function to process your text data

# In[41]:


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[42]:


#test
tokenize("i LOVE kfc!")
#lovely!


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[51]:


def pipeline():
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)), #initial vectorizer to transform data into a 'dictionary'
                    ('tfidf', TfidfTransformer()), #tfidf bit
                    ('clf',MultiOutputClassifier(estimator= RandomForestClassifier())) #classifier that puts the string into one of the multiple(36) cats
                     ])

    
    return pipeline 


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[82]:


#define independent variables and dependent variable
category_names = df[['aid_centers', 'aid_related', 'buildings', 'child_alone', 'clothing','cold', 'death', 'direct_report', 'earthquake', 'electricity', 'fire','floods', 'food', 'hospitals', 'infrastructure_related','medical_help', 'medical_products',  'military','missing_people', 'money', 'offer', 'other_aid','other_infrastructure', 'other_weather', 'refugees', 'related','request', 'search_and_rescue', 'security', 'shelter', 'shops', 'storm','tools', 'transport', 'water', 'weather_related']].columns
X=df[['message']].values


# In[108]:


y=df[category_names]
y.head()


# In[86]:


category_names


# In[140]:


X
#good


# In[128]:


X.shape


# In[131]:


y
#good


# In[101]:


y.shape


# In[44]:


#split the data into test and train, split 30%/70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[77]:


X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0]


# In[78]:


#let's train the pipeline using the built pipeline and training data
model=pipeline()
model.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[22]:


from sklearn.utils.multiclass import type_of_target
type_of_target(y_test)
type_of_target(y_pred)
y_pred = model.predict(X_test)
target_names=np.unique(y_pred)
print(classification_report(np.hstack(y_test),np.hstack(y_pred)))


# In[89]:


from sklearn.utils.multiclass import type_of_target
type_of_target(y_test)
type_of_target(y_pred)
y_pred = model.predict(X_test)
target_names=np.unique(y_pred)
print(classification_report(np.hstack(y_test),np.hstack(y_pred)))


# In[102]:


np.set_printoptions(threshold=sys.maxsize)
ypred=pd.DataFrame(y_pred.tolist())
ypred.sum(axis=1)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[103]:


def pipeline_GS():
    pipeline_GS = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(estimator= RandomForestClassifier()))
                    ])

    parameters = {
                    'vect__min_df':[1,10,50],
                    'tfidf__smooth_idf': [True, False],
                    }
    model  = GridSearchCV(pipeline_GS, param_grid=parameters, cv=3) 
    return model 


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[56]:


model=pipeline_GS()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.utils.multiclass import type_of_target
type_of_target(y_test)
type_of_target(y_pred)
print(classification_report(np.hstack(y_test),np.hstack(y_pred)))
#marginal improvement compared to before, but very good results already


# In[104]:


model=pipeline_GS()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.utils.multiclass import type_of_target
type_of_target(y_test)
type_of_target(y_pred)
print(classification_report(np.hstack(y_test),np.hstack(y_pred)))
#marginal improvement compared to before, but very good results already


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# # I tried to incorporate the starting verb method in the lessons but for some reason keeps getting an 'list index out of range' error. Given that the model is performing very well already i think i'll pass for now. If anyone can share some insight as to how to make this work that would be most helpful

# In[32]:

# In[66]:


def save_model(model, model_filepath):

    with open(model_filepath, 'wb') as pkl_file:
                  pickle.dump(model, pkl_file)
    pkl_file.close()


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[107]:


save_model('model', 'classifier.pkl')

