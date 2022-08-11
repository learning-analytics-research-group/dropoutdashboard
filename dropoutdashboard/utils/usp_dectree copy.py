# Importing packages
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where
import itertools # construct specialized tools
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from . import funcs

#carregando dados
X_train, y_train, X_test, y_test = funcs.tratando_dados()

# Data normalization with sklearn
from sklearn import preprocessing
standardized_X = preprocessing.scale(X_test)


# Transform the dataset
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

# Modelling
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# Predictions
yhat = clf.predict(X_test)
yhat_prob = clf.predict_proba(X_test)

errors, mape, accuracy = funcs.evaluate(lr1, X_test, y_test)
