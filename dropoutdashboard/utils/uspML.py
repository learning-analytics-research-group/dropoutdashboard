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


#carregando dados
dados_usp = pd.read_csv('data/letras_usp_course_completion.csv', sep=',')


def year_max_min():

    min_year = pd.to_datetime(dados_usp['dtaing']).dt.year.min()
    max_year = pd.to_datetime(dados_usp['dtaing']).dt.year.max()

    return (min_year, max_year)

def num_alunos():
    #função que conta numero e alunos não unicos

    num = len(dados_usp['codpes'])

    return num

def evasao():

    evadidos = dados_usp['evadido'].value_counts()['sim']

    return evadidos

def age(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, 
                                      today.day) < (born.month, 
                                                    born.day))

#criando coluna com idade

dados_usp['idade'] = dados_usp['dtanas'].apply(age)

def idade_media():
    return dados_usp['idade'].mean()


