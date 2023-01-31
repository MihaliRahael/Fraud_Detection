# Experimentation to improve Recall metric in fraud detection

## Problem Statement

#### Try to improve Recall metric in a highly imbalanced dataset, for instance given a credit card fraudulent data

## Scenario

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

## Dataset: Overview

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Important: Performance metric

-   In case of fraud detection case studies, predicting a positive point as negative points will cost a lot more than predicting negative points as positive data points.
-   Since the dataset is highly imbalanced 'Accuracy' is not a good metric
-   Highly imbalanced data and cost of errors is so different, so AUC is useless here.
-   Balancing the dataset and use accuracy will solve the issue of data imbalanced but it won’t solve cost of error difference.
-   Both Precision and Recall measure wrt positive points, And F1 score is the harmonic mean of Precision and Recall. These metrices can resolve cost of error problem.

## Data Analysis : EDA and FE

-   There are no missing values in the dataset
-   No need to analyse on outliers since outliers datapoints can be fraudulent
-   Correlation graph





![image](https://user-images.githubusercontent.com/106816732/215701052-d6278d2c-1494-4d24-8d1e-6f5e9cb2fe1b.png)

- Removed Time feature and scaled the dataset using StandardScalar() before training

## Model training and testing results

#### Note : Since there is no noteworthy difference in the F1 scores on training data and testing data, we can confirm that there is no overfitting issue with any model.

Nearest Neighbors
Model Performance of training set
- F1 Score : 0.9995
- Precision : 0.9495
- Recall : 0.7966

Model Performance of test set
- F1 Score : 0.9996
- Precision : 0.9244
- Recall : 0.7971
===================================


Linear SVM
Model Performance of training set
- F1 Score : 0.9993
- Precision : 0.8389
- Recall : 0.7797

Model Performance of test set
- F1 Score : 0.9995
- Precision : 0.8603
- Recall : 0.8478
===================================


RBF SVM
Model Performance of training set
- F1 Score : 0.9996
- Precision : 0.9703
- Recall : 0.8305

Model Performance of test set
- F1 Score : 0.9994
- Precision : 0.9794
- Recall : 0.6884
===================================


Decision Tree
Model Performance of training set
- F1 Score : 1.0000
- Precision : 1.0000
- Recall : 1.0000

Model Performance of test set
- F1 Score : 0.9992
- Precision : 0.7365
- Recall : 0.7899
===================================


Random Forest
Model Performance of training set
- F1 Score : 1.0000
- Precision : 1.0000
- Recall : 1.0000

Model Performance of test set
- F1 Score : 0.9996
- Precision : 0.9652
- Recall : 0.8043
===================================


Neural Net
Model Performance of training set
- F1 Score : 0.9998
- Precision : 0.9734
- Recall : 0.9294

Model Performance of test set
- F1 Score : 0.9996
- Precision : 0.9180
- Recall : 0.8116
===================================


AdaBoost
Model Performance of training set
- F1 Score : 0.9991
- Precision : 0.8369
- Recall : 0.6667

Model Performance of test set
- F1 Score : 0.9992
- Precision : 0.8246
- Recall : 0.6812
===================================


Naive Bayes
Model Performance of training set
- F1 Score : 0.9870
- Precision : 0.0612
- Recall : 0.8164

Model Performance of test set
- F1 Score : 0.9874
- Precision : 0.0600
- Recall : 0.8623
===================================

![image](https://user-images.githubusercontent.com/106816732/215709908-80c8903a-ec39-48bd-b723-ad7ec36e38c5.png)

## Lets try to increase Recall value by customizing softmax threshold and setting class weights
```
# Initialising the ANN
classifier = Sequential()
initializer = tf.keras.initializers.GlorotNormal()
class_weight = {0: 1.,
                1: 3.}

# Adding the input layer and the first hidden layer
classifier.add(Dense(units =15 , kernel_initializer = initializer, activation = 'elu', input_dim = 29))

# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = initializer, activation = 'elu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = initializer, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train_scaled, y_train, batch_size = 32, epochs = 100, class_weight=class_weight)

# Predicting the Test set results
y_pred = classifier.predict(X_test_scaled)

# redefining the softmax threshold
for i, predicted in enumerate(y_pred):
    if predicted[0] > 0.2:
        y_pred[i]=1
    else:
        y_pred[i]=0
```
##### Performance after softmax threshold modification and class balancing
Model Performance of test set after giving class weights and change in softmax threshold
- F1 Score : 0.9992
- Precision : 0.6946
- Recall : 0.8406

![image](https://user-images.githubusercontent.com/106816732/215710999-33b42d27-3914-4e78-bba2-9879e20c2927.png)

## Conclusion
We could see a slight increase in Recall value by making some simple changes in class balancing and redefining the softmax threshold

## Improvements
- We can design a custom error metric like, a|type1 errors| + b|type2 errors|, a>>b.
Here we give different weights to errors. Type 1 error is false positives and type 2 is FN. This custom method will solve data imabalacing using weights and cost of error problem
- The performances may be improved by
  * Hyperparameter tuning
  * Redefining the architecture of ANN
  * Adding Dropouts and Batch normalization
