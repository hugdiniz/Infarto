import numpy
import math
from random import shuffle
from Dataset import lerDataset 
from sklearn.linear_model import SGDClassifier


if __name__ == "__main__":
    matrix = lerDataset()
    lista = list(range(1,len(matrix)))
    shuffle(lista)
    a = numpy.asarray(lista)
    
    treinoIndex = numpy.where(a % 8 != 1)[0]
    testeIndex =  numpy.where(a % 8 == 1)[0]
    
    clf = SGDClassifier(loss="hinge", penalty="l1")
    
    clf.fit([matrix[y,:3] for y in treinoIndex], [matrix[y,3] for y in treinoIndex])
   
    acerto = 0
    for n in testeIndex:
        acerto = acerto + int( matrix[n,3] == clf.predict(matrix[n,:3])[0])
        print("original: "+str(matrix[n,3]) + " estimado: "+str(clf.predict(matrix[n,:3])))
    
    print("Acerto:" +  str(acerto))
    print("porcentagem acerto "+ str( (acerto *100 / len(testeIndex)) ))
    resultado = clf.score([matrix[y,:3] for y in testeIndex], [matrix[y,3] for y in testeIndex])
    print("resultado: "+str(resultado))