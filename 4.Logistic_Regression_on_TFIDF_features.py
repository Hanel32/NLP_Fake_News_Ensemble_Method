################################################################################################
##  Script Info: It Identifies FakeNews using TFidf features and logistic regression classifier  
##  Author: Mohammed Habibllah Baig 
##  Date : 11/22/2017
################################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.linear_model import LogisticRegression

DataFrame= pd.read_csv("./tempdata/Complete_DataSet.csv")
DataFrame.dropna(inplace=True)
### Assignig predictors and target values
bodies = DataFrame.body.values
headlines = DataFrame.headline.values
y = DataFrame.fakeness.values

#clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
#clf.fit(X_train, y_train)
        
####################################################################################
## To iterate over different HyperParameters and check the Accuracy#################
####################################################################################

HyperParameters=[['l1'],[(1,1),(1,2),(1,3)],[0.65,0.75],[0.001,0.01]]

for penalt in HyperParameters[0]:
    for gram in HyperParameters[1]:
        for mx_df in HyperParameters[2]:
            for mn_df in HyperParameters[3]:
                print("For the parameters of: \nmax_df=",mx_df,"min_df=",mn_df,"\nngram_range=",gram,"penalty as=",penalt)
                tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=gram,max_df=mx_df,min_df=mn_df)
                
                X_body_tfidf = tfidf.fit_transform(bodies)
                X_headline_tfidf = tfidf.fit_transform(headlines)

                X_headline_train_tfidf, X_headline_test_tfidf, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
                X_body_train_tfidf, X_body_test_tfidf, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)

                lr = LogisticRegression(penalty=penalt,n_jobs=3)
                
                #lr.fit(X_headline_train_tfidf, y_headline_train)
                #y_pred = lr.predict(X_headline_test_tfidf)

                lr.fit(X_body_train_tfidf, y_body_train)
                y_pred = lr.predict(X_body_test_tfidf)
                
                print("Logistig Regression with TFIDF features F1 and Accuracy Scores : \n")
                #print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_pred, average='macro')*100 ) )
                #print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_headline_test, y_pred)*100) )
                print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_pred, average='macro')*100 ) )
                print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_body_test, y_pred)*100) )