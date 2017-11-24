import sys
import getopt
import os
import math
import operator
import numpy as np
from nltk.corpus import stopwords 

class Maxent:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Maxent initialization"""
    
    self.numFolds     = 10  #Number of times the testing data is folded.
    self.words        = {}  #Dictionary for words in the bag of words.
    self.vocab_length = 0   #Total length of the calculated bag of words
    self.count_docs   = 0   #Count of total documents; just an iterator
    self.bag_of_words = []  #All word occurrences for all documents
    self.bag_of_pos   = []  #All word occurrences for positive 
    self.weights      = []  #Calculated feature weights
    self.accum        = []  #Calculated summation for weight update.
    '''
    * The idea with word_freq is to store at 0 the positive occurences of 
    '''
    self.word_freq    = []
    

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Maxent classifier 

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    words = [word for word in words if word not in stopwords.words('english')]

    weight = 0.
    for w in list(words):
        w = w.lower()
        if w in self.words:
            weight = weight + float(self.weights[int(self.words[w])])

    weight = 1 / (1 + np.exp(-weight))
    #print 'The calculated weight is: ', weight
    if weight > .5:
        return 'pos'
    else:
        return 'neg' 

  def addExample(self, klass, words, doc, eta, lambdaa):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Maxent class.
     * Returns nothing
     *
     * Calculate empirical count for the document
     *   - the empirical count is the sum of the observed occurrences of a classifier in a document
    """
    occurrence = np.zeros(self.vocab_length)
    
    
    klass_int = 0
    if(klass == 'pos'):
        klass_int = 1
    for w in words:
        w = w.lower()
        self.bag_of_words += 1                  #Counts for all words for empirical probability.
        occurrence[int(self.words[w])] = 1
    """
    TODO:
        Add columns for all words in accum; it's the accumulated probability of all prior occurences of the word.
        Add the calculated new weight to accum[word]
        After being able to successfully calculate weights,
        - score a document by adding up weight of all words over the total words seen. If > 50%, it's a positive match
        
        For every word that showed up in the current 
    """
    change = 99
    parse  = 0
    
    #print 'Parsing document: ', doc
    changes = 0
    while(change > eta):
        changes += 1
        if change == 99:
            change = 0
        sum_weights = 0
        for w in set(words):
            w = w.lower()
            sum_weights += self.weights[int(self.words[w])]
        for w in set(words):
            w = w.lower()
            prev_weight = self.weights[int(self.words[w])]
            x_i_j       = occurrence[int(self.words[w])]
            y_i         = klass_int
            lamb_weight = -1 * (lambdaa * prev_weight)
            document_p  = 1 / (1 + np.exp(-sum_weights))
            
            new_weight  = prev_weight + eta*(lamb_weight + x_i_j*(y_i - document_p))
            self.weights[int(self.words[w])] = new_weight
            
            change     += new_weight - prev_weight
            parse      += 1
            
        change          = change / len(set(words))
    #print 'Total changes: ', changes
    pass
  
  def train(self, split, epsilon, eta, lambdaa):
      """
      * TODO 
      * iterates through data examples
      https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
      ^- bag of words optimization
      """

      np.random.shuffle(split.train)

      curr_word = 0
      for example in split.train:
          self.count_docs += 1
          for w in example.words:
              w = w.lower()
              if w not in self.words:
                  self.words[w] = curr_word
                  curr_word += 1
      self.vocab_length = len(self.words.keys())
      print 'The vocab length is: %d' % self.vocab_length + '\n'
      self.bag_of_words = np.zeros(self.vocab_length)
      self.bag_of_pos   = np.zeros(self.vocab_length)
      self.weights      = np.zeros(self.vocab_length)
      
      ex_doc = 0
      for example in split.train:
          words = example.words
          self.addExample(example.klass, words, ex_doc, eta, lambdaa)
          ex_doc += 1
      
      #print 'Weights are now: '
      #for item in self.weights:
          #print item,
      
     
  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits
  

def test10Fold(args):
  pt = Maxent()
  
  splits = pt.crossValidationSplits(args[0])
  epsilon = float(args[1])
  eta = float(args[2])
  lambdaa = float(args[3])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Maxent()
    accuracy = 0.0
    classifier.train(split, epsilon, eta, lambdaa)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
    
    
def classifyDir(trainDir, testDir, eps, et, lamb):
  classifier = Maxent()
  trainSplit = classifier.trainSplit(trainDir)
  epsilon = float(eps)
  eta = float(et)
  lambdaa = float(lamb)
  classifier.train(trainSplit, epsilon, eta, lambdaa)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy
    
def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 5:
    classifyDir(args[0], args[1], args[2], args[3], args[4])
  elif len(args) == 4:
    test10Fold(args)

if __name__ == "__main__":
    main()
