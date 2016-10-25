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
from sklearn.metrics import precision_score,recall_score

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
    matrixTrain = lerXLSX("Banco de Dados - Infarto treino.xlsx",training=True)
    matrixTest = lerXLSX("Banco de Dados - Infarto teste.xlsx")    
   
    X = matrixTrain[:,0]
    Y = matrixTrain[:,1]
    Z = matrixTrain[:,2]
    W = matrixTrain[:,4]
    
    XT = X[np.where(W == 1 )[0]]
    YT = Y[np.where(W == 1 )[0]]
    ZT = Z[np.where(W == 1 )[0]]

    XF = X[np.where(W == -1 )[0]]
    YF = Y[np.where(W == -1 )[0]]
    ZF = Z[np.where(W == -1 )[0]]

    
        
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')


    # ax.scatter(XT, YT, ZT, c="r", marker='o')
    # ax.scatter(XF, YF, ZF, c="b", marker='o')


    # plt.show()

    clf = SVC(kernel="linear", C=0.025)
    
    clf.fit(matrixTrain[:,:4], matrixTrain[:,4])
   
    #acerto = 0
    #for n in testeIndex:
    # #    acerto = acerto + int( matrix[n,3] == clf.predict(matrix[n,:3])[0])
    
    predicts = [clf.predict(matrixTest[n,:4])[0] for n in range(len(matrixTest))]
    score = clf.score(matrixTest[:,:4],matrixTest[:,4])
    print(precision_score(predicts,matrixTest[:,4]))
    print(recall_score(predicts,matrixTest[:,4]))
    print(score)
    