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

    X_overlap = gen_or_load_feats(Jaccard_Similarity, h, b)
    #X_refuting = gen_or_load_feats(refuting_features, h, b)
    X_polarity = gen_or_load_feats(polarity_features, h, b)
    X_hand = gen_or_load_feats(hand_features, h, b)
    X_vader = gen_or_load_feats(sentiment_feature, h, b)
    X_NER = gen_or_load_feats(named_entity_feature, h, b)
    print("X_overlap: ",X_overlap)
    #print("X_refuting: ",X_refuting)
    print("X_polarity: ",X_polarity)
    print("X_hand: ",X_hand,"len:", len(X_hand))
    print("X_vader: ",X_vader)
    print("X_NER:",X_NER)
    X =X_hand+ X_polarity+ X_overlap+X_vader+X_NER
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