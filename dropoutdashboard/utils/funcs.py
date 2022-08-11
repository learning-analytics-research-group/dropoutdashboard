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
from sklearn.metrics import mean_absolute_error

import base64
from collections import Counter
import io
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
import urllib
from sklearn import metrics
from datetime import datetime, date
from matplotlib.backends.backend_agg import FigureCanvasAgg
from django.http import HttpResponse


from collections import Counter


def gerar_imagem(fig):

    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)

    return uri


def tratando_dados():

    #carregando dados
    data = pd.read_csv('data/letras_usp_course_completion.csv', sep=',')

    #transformando target em valor numerico
    data['evadido'] = data['evadido'].apply(lambda x : 1 if (x == 'sim') else 0)

    #seprando dados
    train_data, test_data = train_test_split(data, test_size=0.2)

    #selecionando colunas de features e target

    features = ['idade_no_ingresso',
        'approvals_in_first_year', 'approvals_in_first_year_obrigatorias',
        'failures_in_first_year', 'failures_in_first_year_obrigatorias',
        'approvals_in_first_year_nao_obrigatorias',
        'failures_in_first_year_nao_obrigatorias', 'approvals_in_second_year',
        'approvals_in_second_year_obrigatorias', 'failures_in_second_year',
        'failures_in_second_year_obrigatorias',
        'approvals_in_second_year_nao_obrigatorias',
        'failures_in_second_year_nao_obrigatorias', 'approvals_in_third_year',
        'approvals_in_third_year_obrigatorias', 'failures_in_third_year',
        'failures_in_third_year_obrigatorias',
        'approvals_in_third_year_nao_obrigatorias',
        'failures_in_third_year_nao_obrigatorias', 'auxilios_year1',
        'auxilios_year2', 'auxilios_year3']

    X_train = np.asarray(train_data[features])

    y_train = np.asarray(train_data['evadido'])

    X_test = np.asarray(test_data[features])

    y_test = np.asarray(test_data['evadido'])

    return (X_train, y_train, X_test, y_test)


def evaluate(lr1, x_scaled, y_test):
    predictions = lr1.predict(x_scaled)
    errors = np.mean(abs(predictions - y_test))
    mape = mean_absolute_error(predictions, y_test)*100
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return (errors, mape, accuracy)