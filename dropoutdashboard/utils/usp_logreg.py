from imblearn.over_sampling import SMOTE
from scipy.stats import norm
import pandas as pd # data processing
import numpy as np # working with arrays
from numpy import where
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import itertools # construct specialized tools
import matplotlib.pyplot as plt # visualizations
from matplotlib import rcParams # plot size customization
from sklearn.linear_model import LogisticRegression # model algorithm
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.metrics import jaccard_score as jss # evaluation metric
from sklearn.metrics import precision_score # evaluation metric
from sklearn.metrics import classification_report # evaluation metric
from sklearn.metrics import confusion_matrix # evaluation metric
from sklearn.metrics import log_loss # evaluation metric
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

from collections import Counter

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

def evaluate_reglog():
    errors, mape, accuracy = funcs.evaluate(lr1, X_test, y_test)

    return (errors, mape, accuracy)


errors, mape, accuracy = funcs.evaluate(lr1, X_test, y_test)

