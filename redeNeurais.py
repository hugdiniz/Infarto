import numpy
import math
from random import shuffle
from Dataset import lerDataset 
from pybrain.structure import FeedForwardNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

def splitKFold(y,num_folds):
    
    lista = list(range(1,786))
    shuffle(lista)
    fold_size = int(math.floor(y/num_folds))
    remainder = y-num_folds*fold_size
    groups = numpy.zeros(y)
    cursor = 1
    group = 1

    while(cursor<=y):
        this_fold_size = 0
        if(group <= remainder):
            this_fold_size  = fold_size+1
        else:
            this_fold_size = fold_size
        for j in range(cursor,cursor+this_fold_size-1):
            groups[j] = group
        group += 1
        cursor += this_fold_size
  

    return groups;
  


if __name__ == "__main__":
    matrix = lerDataset()
    
    print(matrix[:,:3])
    trndata = SupervisedDataSet(3,1)
    tstdata = SupervisedDataSet(3,1)
    
    lista = list(range(1,len(matrix)))
    shuffle(lista)
    a = numpy.asarray(lista)
    
    treinoIndex = numpy.where(a % 10 != 1)[0]
    testeIndex =  numpy.where(a % 10 == 1)[0]
    min_max_scaler = preprocessing.MinMaxScaler()
    matrixProcessed = min_max_scaler.fit_transform(matrix[:,:3])
    matrixProcessed = preprocessing.scale(matrixProcessed)
    #print(confusion_matrix(matrixProcessed[:,1],matrixProcessed[:,2]))
    #scores = np.zeros(k)
    #x1Media = np.mean(matrix[:,0])
    #x2Media = np.mean(matrix[:,1])
    #x1STD = np.std(matrix[:,0])
    #x2STD = np.std(matrix[:,1])
    
    
    for n in treinoIndex:
        trndata.addSample(matrixProcessed[n,:3], [matrix[n,3]])
    
    for n in testeIndex:
        tstdata.addSample(matrixProcessed[n,:3], [matrix[n,3]])
    
    fnn = buildNetwork(trndata.indim, 4, trndata.outdim, bias=True)
    
    trainer = BackpropTrainer(fnn, trndata, learningrate=0.1)
    trainer.trainUntilConvergence( verbose = True, maxEpochs = 10)
    trainer.testOnData(tstdata, verbose=True)
    acerto = 0
    for n in testeIndex:
            resultadoInfartoN = fnn.activate(matrixProcessed[n,:3])
            if resultadoInfartoN < 0:
                acerto = acerto + int( matrix[n,3] == -1)
            else:
                acerto = acerto + int( matrix[n,3] == 1)
    print("porcentagem acerto "+ str( (acerto *100 / len(testeIndex)) ))
    
  

    
