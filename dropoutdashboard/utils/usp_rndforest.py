# Importing packages
import numpy as np
from numpy import where
from numpy import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from . import funcs

#carregando dados
X_train, y_train, X_test, y_test = funcs.tratando_dados()

# Data normalization with sklearn


scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Transform the dataset
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train) 

# Modelling
forest = RandomForestClassifier(random_state = 1,
                                  n_estimators = 1000,
                                  max_features = 'sqrt',
                                  max_depth = 50,
                                  bootstrap = False,
                                  min_samples_split = 2,  min_samples_leaf = 1) 
model = forest.fit(X_train, y_train) 
y_pred = model.predict(X_test)



errors, mape, accuracy = funcs.evaluate(model, X_test, y_test)
