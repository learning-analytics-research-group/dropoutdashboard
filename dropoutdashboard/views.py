from django.shortcuts import render
from django.http import HttpResponse

from .utils import logistic_regression, nitraML

def index(request):
    dados = logistic_regression.load()
    return render(request, 'dropoutdashboard/index.html', {'dados': dados })

def participants(request):
    return render(request, 'dropoutdashboard/participants.html')

def sobre(request):
    return render(request, 'dropoutdashboard/sobre.html')

def nitra(request):

    texto = nitraML.shape()
    acuracia = nitraML.acuracia()

    return render(request, 'dropoutdashboard/nitra.html', {'texto' : texto, 
                                                            'acuracia': acuracia,
                                                                            })