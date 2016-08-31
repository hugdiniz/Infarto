import numpy as np
import math
from random import shuffle
from Dataset import lerDataset 
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

if __name__ == "__main__":
    
    k = 8
    matrix = lerDataset()
    lista = list(range(1,len(matrix))) 
    shuffle(lista)
    a = np.asarray(lista) % k   
    scores = np.zeros(k)
    for fold in range(0,8):
        treinoIndex = np.where(a != fold)[0]
        testeIndex =  np.where(a == fold)[0]
        
        clf = GaussianNB()
        
        clf.fit([matrix[y,:3] for y in treinoIndex], [matrix[y,3] for y in treinoIndex])
       
        acerto = 0
        for n in testeIndex:
            acerto = acerto + int( matrix[n,3] == clf.predict(matrix[n,:3])[0])
            #print("original: "+str(matrix[n,3]) + " estimado: "+str(clf.predict(matrix[n,:3])))
        
        print("Acerto:" +  str(acerto))
        print("porcentagem acerto "+ str( (acerto *100 / len(testeIndex)) ))
        scores[fold] = clf.score([matrix[y,:3] for y in testeIndex], [matrix[y,3] for y in testeIndex])
        print("resultado: "+str(scores[fold]))
    
    print(" NB == Score:%2.2e[+/- %2.2e]"%(np.mean(scores), np.std(scores)))