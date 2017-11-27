import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats,sentiment_feature,named_entity_feature
from feature_engineering import Jaccard_Similarity

#Model class wrapper
#Author: Carson Hanel
class Model():
    def __init__(self, datafile, headfile, bodyfile):
        self.body_words   = {}
        self.head_words   = {}
        self.head_weights = []
        self.body_weights = []
        self.data_words   = []
        self.data_headers = []
        self.get_head(headfile)
        self.get_body(bodyfile)
        self.get_data(datafile)
        '''
        Explanation of functions/data:
            - body_words is a dictionary of all words in the body corpus
            - head_words is a dictionary of all words in the headline corpus
            - body_weights is a list of weights of the words for pseudo perceptron
            - head_weights is a list of weights of the words for pseudo perceptron
            - data_words   is the body contents of every file
            - data_headers is the head contents of every file
            - get_head(filename) gets header data for perceptron
            - get_body(filename) gets body   data for perceptron
            - get_data(filename) gets data for feature engineering
        '''
        
    #Grabs all data to be processed.
    def get_data(datafile):
        with open(datafile) as fileName:
            for row in reader.iterrows():
                reader = pd.read_csv(fileName).fillna(value = "")
                head   = row['headline']
                body   = row['body']
                self.data_words.append(body)
                self.data_headers.append(head)
            fileName.close()
    
    #Fills a dictionary of learned weights for body data
    def get_body(bodyfile):
        curr = 0
        with open(bodyfile) as fileName:
            for row in reader.iterrows():
                reader = pd.read_csv(fileName).fillna(value = "")
                word   = row['word']
                weight = row['weight']
                if word not in self.body_words.keys():
                    self.body_words[word]   = curr
                    self.body_weights.append(weight)
                    curr += 1
            fileName.close()
        
    #Fills a dictionary of learned weights for headline data
    def get_head(headfile):
        curr = 0
        with open(headfile) as fileName:
            for row in reader.iterrows():
                reader = pd.read_csv(fileName).fillna(value = "")
                word   = row['word']
                weight = row['weight']
                if word not in self.head_words.keys():
                    self.head_words[word]   = curr
                    self.head_weights.append(weight)
                    curr += 1
            fileName.close()
                
#End current work by Carson on Model class

def generate_features(h,b):

    X_overlap  = gen_or_load_feats(Jaccard_Similarity, h, b)
    X_polarity = gen_or_load_feats(polarity_features, h, b)
    X_hand     = gen_or_load_feats(hand_features, h, b)
    X_vader    = gen_or_load_feats(sentiment_feature, h, b)
    X_NER      = gen_or_load_feats(named_entity_feature, h, b)
    
    #Currently being worked on by Carson:
    X_unaries  = gen_or_load_feats(unaries, h, b)
    X_bigrams  = gen_or_load_feats(bigrams, h, b)
    X_trigrams = gen_or_load_feats(trigrams, h, b)
    X_percept  = gen_or_load_feats(score, h, b)
    #Above is current work
    
    X          =X_hand+ X_polarity+ X_overlap+X_vader+X_NER
    #X = np.concatenate(X,axis=0)
    print(X)
    return X

all_features=[]    
for index, row in df.iterrows():
    features=generate_features(row['headline'],row['body'])
    all_features.append(features)
    
#Data gathering function
#Author: Carson Hanel
def gatherData():
    
np.save("My_features",all_features)
file=open("My_features.txt","w")
file.write(str(all_features))
file.close()

