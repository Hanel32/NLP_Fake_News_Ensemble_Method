import nltk
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from os.path import basename
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import models
from gensim.models.phrases import Phraser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
#from FeatureData import FeatureData, tokenize_text

from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.chunk import tree2conlltags
from nltk.stem import PorterStemmer


_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headline, body):
    feats = feat_fn(headline, body)
    return feats




def Jaccard_Similarity(headline, body):
    X = []
    clean_headline = clean(headline)
    clean_body = clean(body)
    clean_headline = get_tokenized_lemmas(clean_headline)
    clean_body = get_tokenized_lemmas(clean_body)
    features = [
        len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
    X.append(features)
    return features

def sentiment_feature(headline,body):
    sid = SentimentIntensityAnalyzer()
    features = []
    headVader = sid.polarity_scores(headline)
    bodyVader = sid.polarity_scores(body)
    features.append(abs(headVader['pos']-bodyVader['pos']))
    features.append(abs(headVader['neg']-bodyVader['neg']))
    return features
    
def named_entity_feature(headline,body):
    """ Retrieves a list of Named Entities from the Headline and Body.
    Returns a list containing the cosine similarity between the counts of the named entities """
    stemmer = PorterStemmer()
    def get_tags(text):
        return pos_tag(word_tokenize(text))

    def filter_pos(named_tags, tag):
        return " ".join([stemmer.stem(name[0]) for name in named_tags if name[1].startswith(tag)])

    named_cosine = []
    tags = ["NN"]
    
    cosine_simi = []
    head = get_tags(headline)
    body = get_tags(body[:255])

    for tag in tags:
        head_f = filter_pos(head, tag)
        body_f = filter_pos(body, tag)

        if head_f and body_f:
            vect = TfidfVectorizer(min_df=1)
            tfidf = vect.fit_transform([head_f,body_f])
            cosine = (tfidf * tfidf.T).todense().tolist()
            if len(cosine) == 2:
                cosine_simi.append(cosine[1][0])
            else:
                cosine_simi.append(0)
        else:
            cosine_simi.append(0)
    return cosine_simi
    
def refuting_features(headline, body):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    clean_headline = clean(headline)
    clean_headline = get_tokenized_lemmas(clean_headline)
    features = [1 if word in clean_headline else 0 for word in _refuting_words]
    X.append(features)
    return features


def polarity_features(headline, body):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    clean_headline = clean(headline)
    clean_body = clean(body)
    features = []
    features.append(calculate_polarity(clean_headline))
    features.append(calculate_polarity(clean_body))
    X.append(features)
    return features

    

def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headline, body):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = (binary_co_occurence(headline, body)
             + binary_co_occurence_stops(headline, body)
             + count_grams(headline, body))

    print("length:", len(X))
    return X

        #Pseudo perceptron classifier
    #author: Carson Hanel
def score(headline, body, weights, words):
    # Utilizes learned scores from a logistic regression perceptron run on the corpus.
    # The idea is to make the learning quicker by doing learning separately, and simply
    # the learned weights rather than learning and then scoring.
    #
    #TODO:
    #  Utilize the maxent classifier and perceptron both in order to generate weights.
    #  See if accuracy improves if not just classification is included, but the document score.
    weight  = 0
    feature = []    
    #Scoring sequence
    for w in set(body):
        w = w.lower()
        if w in words:
            weight += float(weights[int(words[w])])
    weight = 1 / (1 + np.exp(-(weight))) 
    feature.append(weight)             
    if weight > 0:
        feature.append("1")
    else:
        feature.append("0")
    return feature
                 
def unaries(body, words):
    # Parses the current document, and finds the frequencies of unaries in the bag of words.
    #
    #TODO:
    #  Important note: unless the BoW and weights are generated within Modelling.py, this will be very slow.
    #                  For a better design, I'll be moving the building of the BoW and the weight gathering 
    #                  from the CSV into the other file. This way, we're not loading 130k words/weights into
    #                  the RAM for every file to be scored; it'll be able to be passed as a parameter.
    feature = np.zeros(len(words.keys()))
    body    = body.split()
    for word in body:
        if word in words:
            feature[int(words[word])] += 1
    return feature
