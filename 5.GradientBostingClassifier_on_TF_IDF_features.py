################################################################################################
##  Script Info: It Identifies FakeNews using TFidf features and Gradient Boosting classifier  
##  Author: Mohammed Habibllah Baig 
##  Date : 11/22/2017
################################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from time import time
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

DataFrame= pd.read_csv("./tempdata/Complete_DataSet.csv")
DataFrame.dropna(inplace=True)
### Assignig predictors and target values
X_body_text = DataFrame.body.values
X_headline_text = DataFrame.headline.values
y = DataFrame.fakeness.values

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1,1),max_df=0.65,min_df=0.001)
X_body_tfidf = tfidf.fit_transform(X_body_text)
X_headline_tfidf = tfidf.fit_transform(X_headline_text)

X_headline_train_tfidf, X_headline_test_tfidf, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
X_body_train_tfidf, X_body_test_tfidf, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)


########################################################
## Gradient Boosting Classifier Code ###################
########################################################

clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
clf.fit(X_body_train_tfidf, y_body_train)
y_pred = clf.predict(X_body_test_tfidf)

print("Gradient Boosting classifier with ony TFIDF features : \n")
#print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_pred, average='macro')*100 ) )
#print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_headline_test, y_pred)*100) )
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_body_test, y_pred)*100) )