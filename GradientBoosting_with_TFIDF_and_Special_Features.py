################################################################################################
##  Script Info: It Identifies FakeNews using TFidf features, special Features
##               from Feature Engineering and Gradient Boosting classifier  
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
from feature_engineering_entire_DataSet import  polarity_features, Misc_features, ExtractFeatures, BoW_feature
from feature_engineering_entire_DataSet import Jaccard_Similarity, sentiment_feature, named_entity_feature
from scipy.sparse import hstack

df_all= pd.read_csv("./tempdata/Complete_DataSet.csv")

X_body_text = df_all.body.values
X_headline_text = df_all.headline.values
y = df_all.fakeness.values
print("body size:",len(y))

"""This Function Extracts Features from Feature Engineering and loads it to .npy file"""

def generate_features(h,b):
    print("body size:",len(b))
    print("head size:",len(h))
    X_overlap = ExtractFeatures(Jaccard_Similarity, h, b, "features/overlap."+".npy")
    X_polarity = ExtractFeatures(polarity_features, h, b, "features/polarity."+".npy")
    X_hand = ExtractFeatures(Misc_features, h, b, "features/hand."+".npy")
    X_vader = ExtractFeatures(sentiment_feature, h, b, "features/vader."+".npy")
    X_NER = ExtractFeatures(named_entity_feature, h, b, "features/hand."+".npy")
    X = np.c_[X_hand, X_polarity, X_overlap,X_vader,X_NER]
    return X
    

X_features=generate_features(X_headline_text,X_body_text)
print("The size of special Feature Vector:")
print(np.shape(X_features))

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1,1),max_df=0.65,min_df=0.001)
X_body_tfidf = tfidf.fit_transform(X_body_text)
X_body_tfidf=X_body_tfidf

del df_all

"""Horizontally Combine TFIDF and Special Features"""
X_features=hstack((X_body_tfidf,X_features))

#X_features = np.c_[X_features,X_body_tfidf]
print("The size of Special Feature + TFIDF features:")
print(np.shape(X_features))
X_body_train, X_body_test, y_body_train, y_body_test = train_test_split(X_features,y, test_size = 0.2, random_state=1234)

#clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
clf = LogisticRegression(penalty='l1',n_jobs=3)

clf.fit(X_body_train, y_body_train)
y_pred = clf.predict(X_body_test)

print("Logistic Regression with TFIDF and Feature Engineering: \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_body_test, y_pred)*100) )