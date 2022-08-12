# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler # data normalization
from imblearn.over_sampling import SMOTE
from numpy import where
import itertools # construct specialized tools
from matplotlib import rcParams # plot size customization
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

from collections import Counter

from . import funcs

#carregando dados
X_train, y_train, X_test, y_test = funcs.tratando_dados()

# Transform the dataset
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

# Modelling

svclassifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)
svclassifier.fit(X_train, y_train)


errors, mape, accuracy = funcs.evaluate(svclassifier, X_test, y_test)

