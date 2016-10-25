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
import pickle

from sklearn.cluster import KMeans

if __name__ == "__main__":
   
    warnings.filterwarnings("ignore")      
    matrixTrain = lerXLSX("Banco de Dados - Infarto.xlsx",training=True) 
    clf = SVC(kernel="linear") 
    clf.fit(matrixTrain[:,:4], matrixTrain[:,4])   
    
    with open("clf.pkl",'wb') as f:
        pickle.dump(clf,f)  