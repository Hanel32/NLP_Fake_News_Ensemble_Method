# NLP_Fake_News_Classifier
A naive approach to classifying fake news. The purpose for this assignment is simply to learn what can be derived from classifiers to further research.


Pipeline:
  - Gather data
    - Extract_Data_TheGuardian.py
    - Also extracts from Kaggle Fake News challenge repository
  - Clean data
    - CleaningData_TheGuardian.py
    - Creates the proper CSV file format database
  - Learn perceptron/maximum entropic weights (Current 94 percent accuracy on body, 78 percent on headline)
    - MaxEnt.py     - see parent repository for hyperparameters
    - perceptron.py - see parent repository for hyperparamteres
    - Both classifiers share similar accurracy over headline/body data.
    - All learned data is put into CSV databases. As an example, please see the folders in the repository.
  - Build a feature model
  - Within the model, generate features
  - Run the generated features through a linear classifier.

Note: an aspect of the pipeline is that a classifier is being pre-run on the headline and body data.
      This is because if we were to learn the weights while generating features, the algorithm would
      take more than an hour to fully complete. This is a design choice to favor expedience.
