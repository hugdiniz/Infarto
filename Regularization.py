import numpy as np
import math
from random import shuffle
from Dataset import lerDataset, lerXLSX
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
import sklearn.preprocessing as preprocessing

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    k = 8
    matrix = lerXLSX("Banco de Dados - Infarto treinoTeste1.xlsx")
    lista = list(range(1,len(matrix))) 
    shuffle(lista)
    a = np.asarray(lista) % k   
    scores = np.zeros(k)
    classifiers = [GaussianNB(),SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),SGDClassifier(loss="hinge", penalty="l1")]
    names = ["Naive_bayes","SVM -- linear","SVM -- BMF","Logistic Regression"]
    for x in range(0,len(classifiers)):
        for fold in range(0,8):
            treinoIndex = np.where(a != fold)[0]
            testeIndex =  np.where(a == fold)[0]
            
            clf = classifiers[x]
            X = [matrix[y,:3] for y in treinoIndex]
            X_normalized = preprocessing.normalize(X, norm='l2')
            clf.fit(X, [matrix[y,3] for y in treinoIndex])
           
            acerto = 0
            for n in testeIndex:
                acerto = acerto + int( matrix[n,3] == clf.predict(matrix[n,:3])[0])
                #print("original: "+str(matrix[n,3]) + " estimado: "+str(clf.predict(matrix[n,:3])))
            
            #print("Acerto:" +  str(acerto))
            #print("porcentagem acerto "+ str( (acerto *100 / len(testeIndex)) ))
            scores[fold] = clf.score([matrix[y,:3] for y in testeIndex], [matrix[y,3] for y in testeIndex])
            #print("resultado: "+str(scores[fold]))
    
        print(names[x] + "  == Score:%2.2e[+/- %2.2e]"%(np.mean(scores), np.std(scores)))