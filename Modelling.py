import sys
import string
import pandas as pd
import numpy as np
from feature_engineering import polarity_features, hand_features, gen_or_load_feats,sentiment_feature,named_entity_feature
from feature_engineering import Jaccard_Similarity
from feature_engineering import score

#Model class wrapper
#Author: Carson Hanel
class Model():
    def __init__(self, datafile, headfile, bodyfile):
        self.sourceEncoding = "iso-8859-1"
        self.targetEncoding = "utf-8"
        self.body_words   = {}
        self.head_words   = {}
        self.head_weights = []
        self.body_weights = []
        self.feat_stream  = open("generated_feats.txt", "w")
        self.id_stream    = open("known_identities.txt", "w")
        self.get_head(headfile)
        self.get_body(bodyfile)
        self.process_data(datafile)
        '''
        Explanation of functions/data:
            - body_words is a dictionary of all words in the body corpus
            - head_words is a dictionary of all words in the headline corpus
            - body_weights is a list of weights of the words for pseudo perceptron
            - head_weights is a list of weights of the words for pseudo perceptron
            - get_head(filename) gets header data for perceptron
            - get_body(filename) gets body   data for perceptron
            - get_data(filename) gets data for feature engineering
        '''
        
    #Grabs all data to be processed.
    def process_data(self, datafile):
        print "Data processing module initiated"
        sys.stdout.flush()
        num = 0
        with open(datafile) as fileName:
            reader = pd.read_csv(fileName).fillna(value = " ")
            for index, row in reader.iterrows():
                head     = row['headline']
                head     = ''.join(x for x in head if x in string.printable)
                body     = row['body']
                body     = ''.join(x for x in body if x in string.printable)
                features = generate_features(self, head, body)
                self.feat_stream.write(str(features) + "\n")
                if num % 100 == 0:
                    print "Datafile: " + str(num) + " processed!"
                num += 1
        fileName.close()
    
    #Fills a dictionary of learned weights for body data
    def get_body(self, bodyfile):
        print "Get body module initiated"
        curr = 0
        with open(bodyfile) as fileName:
            reader = pd.read_csv(fileName).fillna(value = " ")
            for index, row in reader.iterrows():
                word   = row['word']
                weight = row['weight']
                #if word not in self.body_words.keys():
                self.body_words[word]   = curr
                self.body_weights.append(weight)
                curr += 1
                if curr % 1000 == 0:
                    print "Parsed: " + str(curr) + " words!"
            fileName.close()
        print "All of body data gathered!"
        
    #Fills a dictionary of learned weights for headline data
    def get_head(self, headfile):
        print "Get head module initiated"
        curr = 0
        with open(headfile) as fileName:
            reader = pd.read_csv(fileName).fillna(value = " ")
            for index, row in reader.iterrows():
                word   = row['word']
                weight = row['weight']
                if word not in self.head_words.keys():
                    self.head_words[word]   = curr
                    self.head_weights.append(weight)
                    curr += 1
            fileName.close()    
        print "All of body data gathered!"
            
#End current work by Carson on Model class
def generate_features(model, h, b):
    X_overlap  = gen_or_load_feats(Jaccard_Similarity, h, b)
    X_polarity = gen_or_load_feats(polarity_features, h, b)
    X_hand     = gen_or_load_feats(hand_features, h, b)
    X_vader    = gen_or_load_feats(sentiment_feature, h, b)
    X_NER      = gen_or_load_feats(named_entity_feature, h, b)
    #Currently being worked on by Carson:
    #X_unaries  = gen_or_load_feats(unaries, b, model.body_words) 
    #X_bigrams  = gen_or_load_feats(bigrams, b, model.body_words) 
    #X_trigrams = gen_or_load_feats(trigrams,b, model.body_words)
    X_body     = score(b, model.body_weights, model.body_words)  
    X_head     = score(h, model.head_weights, model.head_words)
    #X_franken  = score(h, b, model.head_weights, model.body_words)
    #Above is current work
    
    X          = X_hand + X_polarity + X_overlap+X_vader+X_NER
    X         += X_body + X_head    # + X_franken
    #X = np.concatenate(X,axis=0)
    return X

#Idea: Datafile, headfile, bodyfile
model = Model(sys.argv[1], sys.argv[2], sys.argv[3])
