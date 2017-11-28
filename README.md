# NLP_Fake_News_Classifier
A naive approach to classifying fake news. The purpose for this assignment is simply to learn what can be derived from classifiers to further research.


Pipeline:
  - Gather data
  - Clean data
  - Learn perceptron/maximum entropic weights (Current 94 percent accuracy on body, 78 percent on headline)
  - Build a feature model
  - Within the model, generate features
  - Run the generated features through a linear classifier.

Note: an aspect of the pipeline is that a classifier is being pre-run on the headline and body data.
      This is because if we were to learn the weights while generating features, the algorithm would
      take more than an hour to fully complete. This is a design choice to favor expedience.
