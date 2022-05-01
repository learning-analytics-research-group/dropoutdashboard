from django.shortcuts import render
from django.http import HttpResponse

from .utils2 import aux 

def index(request):
    dados = aux.load()
    return render(request, 'dropoutdashboard/index.html', {'dados': dados })