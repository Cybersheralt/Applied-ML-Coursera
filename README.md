# Applied Machine Learning Coursera
> Here lies my assignments for course [Applied Machine Learning in Python](https://www.coursera.org/learn/python-machine-learning).
> There were 4 assignments based on using Python and its libraries for Machine Learning.
> Numeration of assignments responds to numeration on Coursera.


## Table of Contents
* [Assignment 1](#assignment-1)
* [Assignment 2](#assignment-2)
* [Assignment 3](#assignment-3)
* [Assignment 4](#assignment-4)
* [List of used libraries](#list-of-used-libraries)
* [List of used ML models](#list-of-used-ml-models)


## Assignment 1
This assignment is called ***Introduction to Machine Learning*** and its purpose is *"using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients."*.

Dataset was taken from `scikit-learn` library.

```
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
```

Firstly I did some analysis of this dataset and split it for training and testing.
For classification I used k-nearest neighbors classifier with 1 nearest neighbour.
Using this classifier I predicted labels for test dataset and found accuracy of prediction, which was ***91,6%***


## Assignment 2
The goal of this assignment was to *"explore the relationship between model complexity and generalization performance, by adjusting key parameters of various supervised learning models."* .It is divided in 2 parts.

### Part 1 - Regression
For this part next dataset was generated:

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
```

After that, 4 tasks were given.

#### Task 1
For this task I fitted a polynomial `LinearRegression` model on the training data `X_train` for degrees 1, 3, 6, and 9. 
I used `PolynomialFeatures` in `sklearn.preprocessing` to create the polynomial features and then fitted a linear regression model. 
For each model, I found 100 predicted values over the interval `x = np.linspace(0,10,100)` and stored this in a numpy array.

#### Task 2
For this task I fitted a polynomial `LinearRegression` model on the training data `X_train` for degrees 0 through 9. 
For each model I computed the ***R^2*** (coefficient of determination) regression score on the training data as well as the the test data.

#### Task 3
Based on ***R^2*** from previous task I found which degree level leads to underfitting, which to overfitting and what choice of degree level would provide a model with good generalization performance on this dataset.

#### Task 4
For this task I trained two models: a non-regularized `LinearRegression` model (default parameters) and a regularized `Lasso Regression` model (with parameters `alpha=0.01`, `max_iter=10000`) both on polynomial features of degree 12.

### Part 2 - Classification
In part 2 I used [UCI Mushroom Data Set](http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io). 
The data was used to train a model to predict whether or not a mushroom is poisonous.
This part consisted of 3 tasks.

#### Task 1
For this task I trained a `DecisionTreeClassifier` with default parameters and found 5 most important features of mushrooms dataset.

#### Task 2
Firstly I created an `SVC` object with default parameters (i.e. `kernel='rbf', C=1`) and `random_state=0`. 
With this classifier, and the subset of original dataset, I explored the effect of `gamma` on classifier accuracy by using the `validation_curve` function to find the training and test scores for 6 values of `gamma` from `0.0001` to `10` (`np.logspace(-4,1,6)`).
So, for each level of `gamma`, `validation_curve` fitted 3 models on different subsets of the data.

#### Task 3
Based on scores from previous task I found which `gamma` level leads to underfitting, which to overfitting and what choice of `gamma` level would provide a model with good generalization performance on this dataset.


## Assignment 3
This assignment is called ***Evaluation*** and based on training several models and evaluating how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction.
Firstly, I checked percentage of fraud in dataset and it was ***1,641%***

### List of used Classifiers
- Dummy classifier (Obviously, its `accuracy score` was high (***0.9853***), but its `recall score` was ***0.0***)
- `SVC` with default parameters (`accuracy score` - ***0.9908***, `recall score` - ***0.375***, `precision score` - ***1.00***)
- `SVC` with parameters `{'C': 1e9, 'gamma': 1e-07}` (for this classifier i found the confusion matrix)
- Logisitic regression classifier with default parameters (also I created a precision recall curve and a roc curve for this case)
- Logisitic regression classifier with grid search over parameters `'penalty': ['l1', 'l2']`, `'C':[0.01, 0.1, 1, 10, 100]` (used recall for scoring and the default 3-fold cross validation.
 
 
## Assignment 4
This assignment is called ***Understanding and Predicting Property Maintenance Fines*** and based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)).

### Task description
"The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.

The evaluation metric for this assignment is the Area Under the ROC Curve (AUC)."

Firstly I checked the dataset and filled `Null` values in target variable. After that, with help of `GridSearchCV` I found good parameters for my `RandomForestClassifier`, trained it and found needed probabilities.

## List of used libraries
- Pandas
- Numpy
- Scikit-learn

## List of used ML models
- K-nearest neighbors
- Linear regression
- Lasso regression
- Decision Tree Classifier
- Support Vector Classifier
- Dummy classifier
- Logisitic regression
- Random Forest Classifier
