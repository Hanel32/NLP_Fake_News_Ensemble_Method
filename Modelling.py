import pandas as pd
import numpy as np
from feature_engineering import polarity_features, hand_features, gen_or_load_feats,sentiment_feature,named_entity_feature
from feature_engineering import Jaccard_Similarity
from feature_engineering import score, unaries

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
        self.features     = []
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
    def get_data(self, datafile):
        with open(datafile) as fileName:
            reader = pd.read_csv(fileName).fillna(value = "")
            for row in reader.iterrows():
                head   = row['headline']
                body   = row['body']
                self.data_words.append(body)
                self.data_headers.append(head)
            fileName.close()
    
    #Fills a dictionary of learned weights for body data
    def get_body(self, bodyfile):
        curr = 0
        with open(bodyfile) as fileName:
            reader = pd.read_csv(fileName).fillna(value = "")
            for row in reader.iterrows():
                word   = row['word']
                weight = row['weight']
                if word not in self.body_words.keys():
                    self.body_words[word]   = curr
                    self.body_weights.append(weight)
                    curr += 1
            fileName.close()
        
    #Fills a dictionary of learned weights for headline data
    def get_head(self, headfile):
        curr = 0
        with open(headfile) as fileName:
            reader = pd.read_csv(fileName).fillna(value = "")
            for row in reader.iterrows():
                word   = row['word']
                weight = row['weight']
                if word not in self.head_words.keys():
                    self.head_words[word]   = curr
                    self.head_weights.append(weight)
                    curr += 1
            fileName.close()
    
    def save_features(self):
        with open("feature_data.txt", 'w') as datafile:
            for feature in self.features:
                datafile.writerow(feature)
        datafile.close()
        
#End current work by Carson on Model class

#Begin work on generate_features calling function
def do_work(datafile, headfile, bodyfile):
    model = Model(datafile, headfile, bodyfile)
    
    for doc in range(len(model.data_words)):
        model.features.append(generate_features(model, doc))
    model.save_features()
    
#End current work by Carson on the model module

def generate_features(model, doc):
    h = model.data_headers[doc]
    b = model.data_body[doc]
    
    
    X_overlap  = gen_or_load_feats(Jaccard_Similarity, h, b)
    X_polarity = gen_or_load_feats(polarity_features, h, b)
    X_hand     = gen_or_load_feats(hand_features, h, b)
    X_vader    = gen_or_load_feats(sentiment_feature, h, b)
    X_NER      = gen_or_load_feats(named_entity_feature, h, b)
    
    #Currently being worked on by Carson:
    X_unaries  = gen_or_load_feats(unaries, b, model.body_words) 
    #X_bigrams  = gen_or_load_feats(bigrams, b, model.body_words) 
    #X_trigrams = gen_or_load_feats(trigrams,b, model.body_words)
    X_body     = gen_or_load_feats(score, h, b, model.body_weights, model.body_words)  
    X_head     = gen_or_load_feats(score, h, b, model.head_weights, model.head_words)
    #Above is current work
    
    X          = X_hand+ X_polarity+ X_overlap+X_vader+X_NER
    X         += X_unaries + X_body + X_head
    #X = np.concatenate(X,axis=0)
    print(X)
    return X

all_features=[]    
for index, row in df.iterrows():
    features=generate_features(row['headline'],row['body'])
    all_features.append(features)
    
np.save("My_features",all_features)
file=open("My_features.txt","w")
file.write(str(all_features))
file.close()
