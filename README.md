# Experimentation to improve Recall metric in fraud detection

## Problem Statement

\*\* Try to improve Recall metric in a highly imbalanced dataset, for instance given a credit card fraudulent data\*\*

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

## Dataset: Analysis

-   There are no missing values in the dataset
-   No need to analyse on outliers since outliers datapoints can be fraudulent

\`\`\`

df.info()

\<class 'pandas.core.frame.DataFrame'\>

RangeIndex: 284807 entries, 0 to 284806

Data columns (total 31 columns):

\# Column Non-Null Count Dtype

\--- ------ -------------- -----

0 Time 284807 non-null float64

1 V1 284807 non-null float64

2 V2 284807 non-null float64

3 V3 284807 non-null float64

4 V4 284807 non-null float64

5 V5 284807 non-null float64

6 V6 284807 non-null float64

7 V7 284807 non-null float64

8 V8 284807 non-null float64

9 V9 284807 non-null float64

10 V10 284807 non-null float64

11 V11 284807 non-null float64

12 V12 284807 non-null float64

13 V13 284807 non-null float64

14 V14 284807 non-null float64

15 V15 284807 non-null float64

16 V16 284807 non-null float64

17 V17 284807 non-null float64

18 V18 284807 non-null float64

19 V19 284807 non-null float64

20 V20 284807 non-null float64

21 V21 284807 non-null float64

22 V22 284807 non-null float64

23 V23 284807 non-null float64

24 V24 284807 non-null float64

25 V25 284807 non-null float64

26 V26 284807 non-null float64

27 V27 284807 non-null float64

28 V28 284807 non-null float64

29 Amount 284807 non-null float64

30 Class 284807 non-null int64

dtypes: float64(30), int64(1)

memory usage: 67.4 MB

\`\`\`
