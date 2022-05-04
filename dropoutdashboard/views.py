from django.shortcuts import render
from django.http import HttpResponse

from .utils import logistic_regression 

def index(request):
    dados = logistic_regression.load()
    return render(request, 'dropoutdashboard/index.html', {'dados': dados })