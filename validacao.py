import numpy as np
import math
from random import shuffle
from Dataset import lerCSV,lerXLSX
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
import sklearn.preprocessing as preprocessing
import sys 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

if __name__ == "__main__":
   
        
    warnings.filterwarnings("ignore")     
  
    matrixTrain = lerXLSX("Banco de Dados - Infarto treinoTeste.xlsx") 
    clf = SVC(kernel="linear")    
    clf.fit(matrixTrain[:,:4], matrixTrain[:,4])   
        
          
    matrixTest = lerXLSX("Banco de Dados - Infarto validacao.xlsx")
    
    score = clf.score(matrixTest[:,:4],matrixTest[:,4])
    
    acerto = 0
    resultado = np.array([clf.predict(matrixTest[n,:4])[0] for n in range(0,len(matrixTest))])
    infartos = np.where(resultado == 1.0)[0]
    naoInfartos = np.where(resultado == -1.0)[0]
    resultadoT = (np.zeros((1,len(resultado))) + resultado).T   
    
    saida = np.concatenate((matrixTest[:,:4], resultadoT),axis=1)
    np.savetxt("resultado.csv", saida, delimiter=",")
    
    print("Pessoas que tiveram doenca cardiovascular: "+str(len(infartos)))
    print("Pessoas que não tiveram doenca cardiovascular: "+str(len(naoInfartos)))
    print("Acuracia = "+str(score))
    