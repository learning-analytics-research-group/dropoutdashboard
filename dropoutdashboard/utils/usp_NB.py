import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt # visualizations

from . import funcs

#carregando dados
X_train, y_train, X_test, y_test = funcs.tratando_dados()
#Modelling
model = GaussianNB(priors = None, var_smoothing = 1e-09)
# fit the model with the training data
model.fit(X_train,y_train)

# Predictions
predictions = model.predict(X_test)

errors, mape, accuracy = funcs.evaluate(model, X_test, y_test)

