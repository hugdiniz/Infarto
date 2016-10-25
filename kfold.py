import numpy as np
import math
from random import shuffle
from Dataset import lerCSV,lerXLSX
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
from sklearn.metrics import precision_score,recall_score
import redeNeurais

#def precisionRecall(yTest,yScore):
#    positiveScore = np.where(yScore == 1)[0]
#    positiveOk = np.where(yScore == 1 and yScore == yTest)[0]
#    positiveTest = np.where(yTest == 1)[0]
#    precision = len(positiveOk) / len(positiveScore)
#    recall = len(positiveOk) / len(positiveTest) 
#    return precision, recall
    

def KFold(matrix, kmeans = True,metric = ("svm linear",SVC(kernel="linear", C=0.025))):
   
    if kmeans:
        limit = 4
    else:
        limit = 3
    k = 8
    lista = list(range(1,len(matrix))) 
    shuffle(lista)
    a = np.asarray(lista) % k   
    
    scores = np.zeros(k)
    recall = np.zeros(k)
    precision = np.zeros(k)
    average_precision = np.zeros(k)
    
    for fold in range(0,k):
        treinoIndex = np.where(a != fold)[0]
        testeIndex =  np.where(a == fold)[0]        
        
        clf = metric[1]        
        clf.fit([matrix[y,:limit] for y in treinoIndex], [matrix[y,4] for y in treinoIndex])       
        
        predicts = clf.predict([matrix[y,:limit] for y in testeIndex]) 
        testOuts = [matrix[y,4] for y in testeIndex]     
       
        scores[fold] = clf.score([matrix[y,:limit] for y in testeIndex],[matrix[y,4] for y in testeIndex] )
        precision[fold] = precision_score(testOuts, predicts)
        recall[fold] = recall_score(testOuts, predicts)

    print("Final:"+ metric[0] + " = Score:%2.2e[+/- %2.2e]"%(np.mean(scores),np.std(scores))) 
    print("Final:"+ metric[0] + " = recall:%2.2e[+/- %2.2e]"%(np.mean(recall),np.std(recall)))
    print("Final:"+ metric[0] + " = precision:%2.2e[+/- %2.2e]"%(np.mean(precision),np.std(precision)))            
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    k = 8
    matrix = lerXLSX("Banco de Dados - Infarto treino.xlsx",training=True)
    
    metrics = [("Naive Bayes",GaussianNB()),("Logistic Regression",SGDClassifier())]
    tmeans = dict()
    tstds = dict()
    
    fmeans = dict()
    fstds = dict()

    # for metric in metrics:        
    #     tKfold = KFold(matrix,True,metric)


    repetition = 5
    
    for metric in metrics:        
        tKfold = KFold(matrix,True,metric) 