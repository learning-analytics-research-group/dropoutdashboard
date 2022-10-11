import base64
import io
from django.shortcuts import render
from django.http import HttpResponse
import urllib

from .utils import logistic_regression, nitraML, uspML, usp_logreg, usp_dectree, usp_rndforest, usp_NB, usp_SVM , usp_NN

def index(request):
    dados = logistic_regression.load()
    return render(request, 'dropoutdashboard/index.html', {'dados': dados })

def participants(request):
    return render(request, 'dropoutdashboard/participants.html')

def sobre(request):
    return render(request, 'dropoutdashboard/sobre.html')

def nitra(request):

    texto             = nitraML.shape()
    acuracia          = nitraML.acuracia()
    fig               = nitraML.scatter_pre_data_augmentation()
    fig2              = nitraML.scatter_pos_data_augmentation()
    confunsion_matrix = nitraML.cnf_matrix()
    roc_auc           = nitraML.roc_auc_plot()
    aprovados, evasao = nitraML.counter_v()
    

    return render(request, 'dropoutdashboard/nitra.html', {'texto'       : texto, 
                                                            'acuracia'   : acuracia,
                                                            'fig'        : fig,
                                                            'fig2'       : fig2,
                                                            'cnf_matrix' : confunsion_matrix,
                                                            'roc_auc'    : roc_auc,
                                                            'aprovados'  : aprovados,
                                                            'evasao'     : evasao,

                                                                            })

def usp(request):
    num_alunos         = uspML.num_alunos() 
    evadidos           = uspML.evasao() 
    indice_evasao      = (evadidos / num_alunos) * 100
    idade_media        = uspML.idade_media()
    fig1               = uspML.evasao_failure1ano_pie(1)
    fig2               = uspML.evasao_failure1ano_bar()
    fig3               = uspML.idade_histograma()
    accuracy_log       = usp_logreg.accuracy
    accuracy_dectree   = usp_dectree.accuracy 
    accuracy_rndforest = usp_rndforest.accuracy
    accuracy_NB        = usp_NB.accuracy
    accuracy_SVM       = usp_SVM.accuracy
    accuracy_NN        = usp_NN.accuracy
    errors_log       = usp_logreg.errors
    errors_dectree   = usp_dectree.errors 
    errors_rndforest = usp_rndforest.errors
    errors_NB        = usp_NB.errors
    errors_SVM       = usp_SVM.errors
    errors_NN        = usp_NN.errors
    mape_log       = usp_logreg.mape
    mape_dectree   = usp_dectree.mape 
    mape_rndforest = usp_rndforest.mape
    mape_NB        = usp_NB.mape
    mape_SVM       = usp_SVM.mape
    mape_NN        = usp_NN.mape

    return render(request, 'dropoutdashboard/usp.html', {
                                                        'num_alunos'          : f'{num_alunos:,}'.format(),
                                                        'evadidos'            : f'{evadidos:,}'.format(),
                                                        'indice_evasao'       : f'{(indice_evasao):.2f}'.format(),
                                                        'idade_media'         : f'{idade_media:.2f}'.format(),
                                                        'fig1'                : fig1,
                                                        'fig2'                : fig2,
                                                        'fig3'                : fig3,
                                                        'accuracy_log'        : '{:0.2f}%'.format(accuracy_log),
                                                        'accuracy_dectree'    : '{:0.2f}%'.format(accuracy_dectree),
                                                        'accuracy_rndforest'  : '{:0.2f}%'.format(accuracy_rndforest),
                                                        'accuracy_NB'         : '{:0.2f}%'.format(accuracy_NB),
                                                        'accuracy_SVM'        : '{:0.2f}%'.format(accuracy_SVM),
                                                        'accuracy_NN'         : '{:0.2f}%'.format(accuracy_NN),
                                                        'errors_log'          : '{:0.3f} degrees'.format(errors_log),
                                                        'errors_dectree'      : '{:0.3f} degrees'.format(errors_dectree), 
                                                        'errors_rndforest'    : '{:0.3f} degrees'.format(errors_rndforest),
                                                        'errors_NB'           : '{:0.3f} degrees'.format(errors_NB),
                                                        'errors_SVM'          : '{:0.3f} degrees'.format(errors_SVM),
                                                        'errors_NN'           : '{:0.3f} degrees'.format(errors_NN),
                                                        'mape_log'            : '{:0.2f}%'.format(mape_log),
                                                        'mape_dectree'        : '{:0.2f}%'.format(mape_dectree),
                                                        'mape_rndforest'      : '{:0.2f}%'.format(mape_rndforest),
                                                        'mape_NB'             : '{:0.2f}%'.format(mape_NB),
                                                        'mape_SVM'            : '{:0.2f}%'.format(mape_SVM),
                                                        'mape_NN'             : '{:0.2f}%'.format(mape_NN),

                                                        
                                                        })

                                                                       