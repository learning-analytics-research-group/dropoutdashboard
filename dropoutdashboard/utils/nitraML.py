import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/DBS.csv', sep=';')

def shape():

    linha, coluna = train_data.shape

    return f'Este dataset possui {linha} linhas e {coluna} colunas.'

def acuracia():
    return '50%'