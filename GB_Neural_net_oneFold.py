################################################################################################
##  Script Info: It Identifies FakeNews using Neural network of weights from Maxent Classifier
##               and extracted features from feature_engineering.py  
##  Author: Mohammed Habibllah Baig 
##  Date : 11/22/2017
################################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split ,StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import re
from scipy.sparse import hstack

df_all= pd.read_csv("./tempdata/Complete_DataSet.csv")

y = df_all.fakeness.values

X_features=[]
for line in open('./tempdata/generated_feats_HFS_full.txt'):
    feat=[]
    feat=line.rstrip().split(',')
    X_features.append(feat)

X_body_train, X_body_test, y_body_train, y_body_test = train_test_split(X_features,y, test_size = 0.2, random_state=1234)

clf = GradientBoostingClassifier(n_estimators=400, random_state=14128, verbose=True)
#clf = LogisticRegression(penalty='l1',n_jobs=3)

clf.fit(X_body_train, y_body_train)
y_pred = clf.predict(X_body_test)

print("Gradient Boosting with Neural Network : \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_body_test, y_pred)*100) )