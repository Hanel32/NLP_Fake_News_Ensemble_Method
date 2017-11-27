import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats,sentiment_feature,named_entity_feature
from feature_engineering import Jaccard_Similarity

df= pd.read_csv("Complete_DataSet_removed_single_letters.csv")
df=df[:10]
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
    
np.save("My_features",all_features)
file=open("My_features.txt","w")
file.write(str(all_features))
file.close()
