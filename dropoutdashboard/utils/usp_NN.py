import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from numpy import where
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import np_utils
from keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from collections import Counter

from . import funcs

#carregando dados
X_train, y_train, X_test, y_test = funcs.tratando_dados()

standardized_X_train = preprocessing.scale(X_train)
standardized_X_test = preprocessing.scale(X_test)

# Transform the dataset
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(standardized_X_train, y_train)

# Define the Keras model
classifier = Sequential()


classifier.add(Dense(10, activation='relu', kernel_initializer='random_normal', input_dim=22))

classifier.add(Dense(5, activation='relu', kernel_initializer='random_normal'))

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# Compile the Keras model
classifier.compile(optimizer ='adam', loss='mean_squared_error', metrics =['accuracy'])

# Fit the Keras model on the dataset
#history = classifier.fit(X_train, y_train, batch_size=5, epochs=100, validation_data=(standardized_X_test,y_test))

errors, mape, accuracy = funcs.evaluate(classifier, standardized_X_test, y_test)