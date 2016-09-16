import numpy as np
import math
from random import shuffle
from Dataset import lerDataset,lerXLSX
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
    #treinoPath = sys.argv[1]
    #testePath = sys.argv[2]
    warnings.filterwarnings("ignore")
    k = 8
    matrixTrain = lerXLSX("Banco de Dados - Infarto treino.xlsx")    

    X = matrixTrain[:,0]
    Y = matrixTrain[:,1]
    Z = matrixTrain[:,2]
    W = matrixTrain[:,3]

    XT = X[np.where(W == 1 )[0]]
    YT = Y[np.where(W == 1 )[0]]
    ZT = Z[np.where(W == 1 )[0]]

    XF = X[np.where(W == -1 )[0]]
    YF = Y[np.where(W == -1 )[0]]
    ZF = Z[np.where(W == -1 )[0]]

    est = KMeans(n_clusters=4, n_init=1, init='random')
    est.fit(matrixTrain[:,:3])
    
    T = np.array([ est.predict(matrixTrain[x,:3])[0] for x in range(0,len(matrixTrain))])
        
    class1 = np.where(T == 0 )[0]
    class2 = np.where(T == 1 )[0]
    class3 = np.where(T == 2 )[0]
    class4 = np.where(T == 3 )[0]
    
    # mT = np.where(W == 1 )[0]
    # fT = np.where(W == -1 )[0]
    
    class1T = np.where(W[class1] == 1 )[0]
    class2T = np.where(W[class2] == 1 )[0]
    class3T = np.where(W[class3] == 1 )[0]
    class4T = np.where(W[class4] == 1 )[0]
    
    class1F = np.where(W[class1] == -1 )[0]
    class2F = np.where(W[class2] == -1 )[0]
    class3F = np.where(W[class3] == -1 )[0]
    class4F = np.where(W[class4] == -1 )[0]

    p1 = float(len(class1T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))
    print("Porcentagem de infartos da classe1:"+ str(p1))
    
    p2 = float(len(class2T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))
    print("Porcentagem de infartos da classe2:"+ str(p2))
    
    p3 = float(len(class3T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))
    print("Porcentagem de infartos da classe3:"+ str(p3))
    
    p4 = float(len(class4T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))
    print("Porcentagem de infartos da classe4:"+ str(p4))
    
    # V  = np.zeros(len(W))
    
    # V[class1T] = p1
    # V[class1F] = 1 - p1
    
    # V[class2T] = p2
    # V[class2F] = 1 - p2
    
    # V[class3T] = p3
    # V[class3F] = 1 - p3
        
    # V[class4T] = p4
    # V[class4F] = 1 - p4
    
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')


    # ax.scatter(XT, YT, ZT, c="r", marker='o')
    # ax.scatter(XF, YF, ZF, c="b", marker='o')


    # plt.show()

    # clf = SVC(kernel="linear", C=0.025)
    
    # clf.fit(matrixTrain[:,:3], matrixTrain[:,3])
   
    # #acerto = 0
    # #for n in testeIndex:
    # #    acerto = acerto + int( matrix[n,3] == clf.predict(matrix[n,:3])[0])
    
    # score = clf.score(matrixTest[:,:3],matrixTest[:,3])
    # print(score)
    