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
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train) 

# Modelling
lr1 = LogisticRegression(max_iter=1000)
lr1.fit(X_train_res, y_train_res)

# Predictions
yhat = lr1.predict(X_test)
yhat_prob = lr1.predict_proba(X_test)

