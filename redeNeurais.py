import math
from random import shuffle
from pybrain.structure import FeedForwardNetwork
from pybrain.datasets            import SupervisedDataSet,ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from sklearn import preprocessing
import numpy as np
import pickle

class RedeNeural(object):
    
    def __init__(self, neuronios = 4):
        self.neuronios = neuronios      

    def fit(self,trainMatrix, testMatrix):
        
        trainMatrix = np.array(trainMatrix)
        limite = len(trainMatrix[0,:])
        min_max_scaler = preprocessing.MinMaxScaler()
        
        if limite == 4:
            feature4 = (np.zeros((1,len(trainMatrix))) + trainMatrix[:,3])
            feature4 = feature4.T
            matrixProcessed = np.concatenate((min_max_scaler.fit_transform(trainMatrix[:,:3]),feature4),axis=1)
        else:
            matrixProcessed = min_max_scaler.fit_transform(trainMatrix)
        
        trainMatrix = np.array(trainMatrix)
        testMatrix = np.array(testMatrix)
        

       
        trndata = ClassificationDataSet(limite,nb_classes=2)
                
        self.fnn = buildNetwork(trndata.indim, self.neuronios, trndata.outdim)
        
        for n in range(0,len(matrixProcessed)):
            trndata.addSample(matrixProcessed[n,:], [testMatrix[n]])       
        
        self.trainer = BackpropTrainer(self.fnn, trndata, learningrate=0.2)
        self.trainer.trainUntilConvergence(maxEpochs = 20)
    
    def score(self,X, Y):

        X = np.array(X)
        Y = np.array(Y)
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        acerto = 0 
        for n in range(0,len(X)):    
            predicao = self.fnn.activate(X[n,:])  
            if predicao < 0:
                acerto = acerto + int( Y[n] == -1)
            else:
                acerto = acerto + int( Y[n] == 1)
            
        
        return acerto / len(X)
    
    def predict(self,X):
        X = np.array(X)
        
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        acerto = 0 
        predicao = self.fnn.activate(X)  
        if predicao < 0:
            return 0
        else:
            return 1
        
