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
    
    for n in treinoIndex:
        trndata.addSample(matrix[n,:3], [matrix[n,3]])
    
    for n in testeIndex:
        tstdata.addSample(matrix[n,:3], [matrix[n,3]])
    
    fnn = buildNetwork(trndata.indim, 4, trndata.outdim, bias=True)
    
    trainer = BackpropTrainer(fnn, trndata, learningrate=0.6, momentum=0.99)
    trainer.trainUntilConvergence( verbose = True, maxEpochs = 1000)
    trainer.testOnData(tstdata, verbose=True)
    
    for n in testeIndex:
        print("original: "+str(matrix[n,3]) + " estimado: "+str(fnn.activate(matrix[n,:3])))
    
    p = fnn.activateOnDataset( tstdata )
    a = numpy.asarray(p)
    numpy.savetxt("foo.csv", a, delimiter=",")

    
