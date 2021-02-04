# Debias and Sentiment Analysis
My objective is to design a machine learning algorithm using classifiers to predict the pre-defined set of topics (Gender, Age) in the gender based datasets. Also to reduce the gender and age bias in and reduce the discrimination based on gender and age. 

# Data sets
- https://www.kaggle.com/crowdflower/twitter-user-gender-classification
- http://archive.ics.uci.edu/ml/datasets/Adult
- https://www.kaggle.com/rtatman/blog-authorship-corpus

# Feature Design
1. Term Frequency: summarizes how often the word appears in a document.
2. Inverse Document Frequency: Downscales words which appear a lot many times in a document.

# Algorithms

Algorithms:
1. Multinomial Naïve Bayes - Naïve classification method is based on Bayes rule. The advantage of Naïve Bayes is it works well on extremely large features and it rarely over fits the data. It is very fast in training and prediction for the amount of data it can handle
2. Random forest - It is a classification algorithm consisting of many decision trees. Each tree in a random forest spits out a class prediction and class with most votes becomes model predictions.
3. SVM - SVM algorithm determines the best decision boundary between vectors that belong to a given class and vectors that do not belong to it.
4. Voting Classifier using SVM - A voting classifier model combines multiple models into a single model which is stronger than individual models. The Ensemble voting classifier uses the arg max function which predicts the classes sharply.
5. Multi layer perceptron - A multilayer perceptron (MLP) is a deep, artificial neural network.They are composed of an input layer to receive the signal, an output layer that makes a decision or prediction about the input, and in between those two, an arbitrary number of hidden layers that are the true computational engine of the MLP.

# Methodology

The training set is split into 80% of training data and 20% as validation data set with which the learning of the classifiers took place and then the whole test data is passed to the classifier for predictions.

# Hyper Parameters

| Classifiers | Tweet Data | Adult Data | Age Data|
| ------ | ------ |------|------|
|Naive Bayes  | alpha=1.28101|alpha=41|alpha=0.02|
| SVM | C=1.0,gamma=1.3|C=1.0,gamma=1.3|default|
| MLP| hidden_layer_sizes=15,max_iter=2500|hidden_layer_sizes=15,max_iter=2000| default|
|Random Forest Classifier | n_estimators = 18,random_state = 42 |n_estimators= 90,min_samples_split= 2,min_samples_leaf= 2,max_features='auto',max_depth=None,bootstrap= True|n_estimators= 80,min_samples_split= 5, min_samples_leaf= 1, max_features='sqrt', max_depth=70,bootstrap= False|
| Voting | Voting=hard |Voting=hard|Voting=hard|


# Classifier Accuracies (Twitter data set)
| Classifiers | Actual Male/Female Counts| Before Bias Male/Female Counts |After Bias Male/Female Counts | Before Bias Accuracy (%)| Training Accuracy (%)| Validation Accuracy/After Bias accuracy (%)
| ------ | ------ |------|------|------| -------| ------|
|Random Forest| Female-572 <br> Male-989 |Female-504 <br> Male-1057| Female-603 <br> Male-958| 68.28|95.16|68.98|
Multinomial Naïve Bayes| Female-572 <br>Male-989|Female-609 <br>Male-952|Female-610 <br>Male-951|67.26|73.59|67.54|
SVM|Female-572 <br>Male-989|Female-173 <br>Male-1388|Female-224 <br>Male-1337|66.75|95.65|66.99|
Voting|Female-572 <br>Male-989|Female-418 <br>Male-1143|Female-505 <br>Male-1056|69.76|89.96|70.36|
Multi layer perceptron|Female-572 <br>Male-989 |Female-180 <br>Male-1381|Female-220 <br>Male-1341|68.20|95.46|68.72|
# Classification Accuracies: ( Adult data set)
|Classifiers | Training Accuracy (%) | Validation Accuracy (%) |
| ------ | ------ |------|
Random Forest |74.50|72.71|
Multinomial Naïve Bayes|71.74|71.28|
SVM|73.66|72.77
Multi layer perceptron|74.01|72.83
Voting|73.47|73.01
# Classification Accuracies: ( Corpus Age data set)
|Classifiers | Training Accuracy (%) | Validation Accuracy (%) |
| ------ | ------ |------|
Random Forest|69.67|73.91
Multinomial Naïve Bayes|57.34|60.72
Voting|64.23|67.60

# Pros and Cons
The pros and cons of different classifiers are discussed in the table below:
|Classifiers| Advantages| Disadvantages|
|------|------|------|
Naïve Bayes| - Simple to design <br> - Easy to tune hyperparameters <br> - Faster performance | Due to simplicity they are often beaten by models properly trained and tuned|
SVM|- Fairly robust to overfitting in high dimensional space <br> - There are many kernels to choose | - Memory intensive <br> - Tuning is difficult and tough to pick the right kernel|
MLP| - Distinguish data that is not linearly separable <br> -  Used for the regression and mapping | - Too many parameters because it is fully connected <br> - Nodes are connected in a very dense web resuling in redundancy and inefficiency|
Random Forest Classifier|- Performs well using a small number of samples <br> - Computational cost is low |- Very little control on the model<br> - Overfitting can easily occur|
Voting Classifier| -Produces better predictions - Robust, able to model nonlinear decision boundaries | - Computationally intensive to train <br> - Prone to overfitting|

# Ideas of Improvement

1. The training set should be a larger one.
2. More usage of ensemble methods such as bagging, boosting and staking.
3. More classifiers should be used in voting classifier.
4. N-grams has to be used in pre-processing instead of one gram/bigram

# Language used 
python, machine learning







  
  

  
