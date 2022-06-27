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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error
from imblearn.over_sampling import SMOTE

#carregando bases de treino e teste
train_data = pd.read_csv('data/DBS.csv', sep=';')
test_data = pd.read_csv('data/DBS_2020.csv', sep=';')

#criando arrays com os dados de treino
X_train = np.asarray(train_data[['access', 'tests', 'assignments']])
y_train = np.asarray(train_data['graduate'])

#criando arrays com os dados de teste
X_test = np.asarray(test_data[['access', 'tests', 'assignments']])
y_test = np.asarray(test_data['graduate'])

#Realizando processo de data augmentation - criação de dados sintéticos
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Modelling
lr1 = LogisticRegression()
lr1.fit(X_train_res, y_train_res)

#realizando a predição sobre o conjunto de teste
y_pre = lr1.predict(X_test)
y_pred_proba = lr1.predict_proba(X_test)[::,1]


def shape():

    linha, coluna = train_data.shape

    return f'Este dataset possui {linha} linhas e {coluna} colunas.'

def counter_v():

    aprovados, evasao = train_data['graduate'].value_counts().values

    return (aprovados, evasao)


def acuracia():
    predictions = lr1.predict(X_test)
    errors = abs(predictions - y_test)
    mape = mean_absolute_error(predictions, y_test)*100
    accuracy = 100 - mape
    return '{:0.2f}%'.format(accuracy)



def scatter_pre_data_augmentation():
    fig, ax = plt.subplots(figsize=(9,5))
    counter = Counter(y_train)

    

    for label, _ in counter.items():
        row_ix = np.where(y_train == label)[0]
        ax.scatter(X_train[row_ix, 0], X_train[row_ix, 1], label=str(label))
    plt.legend()
    plt.xlabel('Access')
    plt.ylabel('Tests')
    plt.title('Distribuição de features por classe do conjunto de treino')

    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)

    

    return uri

#utilizando variaveis de entrada na função, poderiamos plotar os dois graficos com a mesma função.
def scatter_pos_data_augmentation():
    fig, ax = plt.subplots(figsize=(9,5))
    counter = Counter(y_train)

    

    for label, _ in counter.items():
        row_ix = np.where(y_train_res == label)[0]
        ax.scatter(X_train_res[row_ix, 0], X_train_res[row_ix, 1], label=str(label))
    plt.legend()
    plt.xlabel('Access')
    plt.ylabel('Tests')
    plt.title('Distribuição de features por classe do conjunto de treino - Pos Data Augmentation')

    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)

    return uri


def plot_confusion_matrix(cm, classes,

                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def cnf_matrix():
    cnf_matrix = confusion_matrix(y_test, y_pre)

    print("Recall metric in the testing dataset: {:0.2f}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
    
    class_names = [0,1]
    fig = plt.figure(figsize=(9,5))
    plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')

    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)

    return uri

def roc_auc_plot():
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    fig = plt.figure(figsize=(9,5))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(fpr,tpr,label="Logistic regression, AUC="+str(auc))
    plt.legend(loc=4)
    
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)

    return uri
