import numpy as np
import math
from random import shuffle
from Dataset import lerXLSX
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
import sklearn.preprocessing as preprocessing
import sys 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from sklearn.metrics import average_precision_score

from sklearn.cluster import KMeans

def precisionRecall(yTest,yScore):
    positiveScore = np.where(yScore == 1)[0]
    positiveTest = np.where(yTest == 1)[0]
    positiveOk = np.where(np.logical_and((yScore == 1),(yTest == 1)))[0]
        
    precision = len(positiveOk) / len(positiveScore)
    recall = len(positiveOk) / len(positiveTest)
    
    print("precision = "+str(precision))
    print("recall = " +str(recall)) 
    
    return precision, recall
    

if __name__ == "__main__":
   
    warnings.filterwarnings("ignore")
    k = 8
    matrixTrain = lerXLSX("Banco de Dados - Infarto.xlsx")
    matrixTest = lerXLSX("Banco de Dados - Infarto teste.xlsx")


    Y = matrixTrain[:,1] 
    YPositivo = matrixTrain[np.where(matrixTrain[:,4] == 1),1]  
    
    totalIdade = [(x,len(np.where(x == Y)[0])) for x in np.unique(matrixTrain[:,1])]
    totalPIdade = np.array([[len(np.where(x == YPositivo)[0]),x] for x in np.unique(matrixTrain[:,1])])

    print(totalIdade)
    print("\n\n")
    print(totalPIdade)
    np.savetxt('total_de_Pidade.csv', totalPIdade, delimiter=',')
    plt.hist(totalPIdade, np.unique(matrixTrain[:,1]), histtype='bar', rwidth=0.8)
    plt.show()


    