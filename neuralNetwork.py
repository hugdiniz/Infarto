from Dataset import lerCSV,lerXLSX
import warnings
import sys 
import numpy as np
import pickle
from sklearn.cluster import KMeans
import theanets
from sklearn import preprocessing
from random import shuffle
from sklearn.metrics import precision_score,recall_score
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


k = 10


if __name__ == "__main__":
    warnings.filterwarnings("ignore")   
    matrix = lerXLSX("Banco de Dados - Infarto treinoTeste.xlsx",training=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
   
    #infartoIdade = np.array([[idade,sum([int(matrix[idPerson,4] == 1) for idPerson in np.where(matrix[:,1] == idade)[0]])] for idade in range(int(np.max(matrix[:,1]))+1)])
    #infartoAgrupado = np.array([idade,])
    #matrixInfarto = np.array(matrix[np.where(matrix[:,4] == 1)[0],:])
    #print(matrixInfarto[1,:])
    #ax.scatter(matrixInfarto[:,0], infartoIdade[np.array(matrixInfarto[:,1],dtype="int32"),1], matrixInfarto[:,2], c='r', marker='o')

    #nInfartoIdade = np.array([[idade,sum([int(matrix[idPerson,4] == -1) for idPerson in np.where(matrix[:,1] == idade)[0]])] for idade in range(int(np.max(matrix[:,1]))+1)])
    
    #matrixNInfarto = np.array(matrix[np.where(matrix[:,4] == -1)[0],:])
    #ax.scatter(matrixNInfarto[:,0], infartoIdade[np.array(matrixNInfarto[:,1],dtype="int32"),1], matrixNInfarto[:,2], c='b', marker='o')
    #plt.show()
    
    #matrixInfarto = np.array(inputs[np.where(outputs == 1)[0],:])
    #infartoIdade = np.array([[idade,sum([int(outputs[idPerson] == 1) for idPerson in np.where(inputs[:,1] == idade)[0]])] for idade in range(int(87))])
    #idadeDescretizada  = min_max_scaler.fit_transform(infartoIdade[np.array(inputs[:,1],dtype="int32"),1])
    min_max_scaler = preprocessing.MinMaxScaler()
    matrixProcessed = min_max_scaler.fit_transform(matrix[:,:3])    
   
    lista = list(range(1,len(matrix))) 
    shuffle(lista)
    a = np.asarray(lista) % k
    for fold in range(0,k):
        treinoIndex = np.where(a != fold)[0]
        testeIndex =  np.where(a == fold)[0]
        
        inputs = np.array([[matrixProcessed[y,0],matrixProcessed[y,1],matrixProcessed[y,2]]  for y in treinoIndex])        
        outputs = np.array([int(matrix[y,4] == 1) for y in treinoIndex],dtype="int32")        
        weights = np.array([int(matrix[y,4] == 1)*2+1 for y in treinoIndex],dtype="int32")  
        model = theanets.Classifier([3, (7, 'tanh'), 2],loss='CrossEntropy',weighted=True)   
       
        model.train([inputs, outputs,weights],algo='sgd',learning_rate=0.01,momentum=0.9)
        #for tm, _ in model.itertrain([inputs, outputs],algo='nag',learning_rate=0.01):
        #    print(tm['loss'])
        with open("model.pkl",'wb') as f:
            pickle.dump(model,f)


        testInputs = np.array([matrixProcessed[y,:3]  for y in testeIndex])               
        predicts = model.predict(testInputs)

        testOuts = np.array([int(matrix[y,4] == 1) for y in testeIndex],dtype="int32")

       

        print("Recall:")
        print(recall_score(testOuts, predicts))  
        print("Precision:")
        print(precision_score(testOuts, predicts))

        score = sum([int(predicts[x] == testOuts[x]) for x in range(len(predicts))]) / len(predicts) 
        print("score:")
        print(score)

        matrixInfarto = np.array(testInputs[np.where(predicts == 1)[0],:])      
        ax.scatter(matrixInfarto[:,0], matrixInfarto[:,1], matrixInfarto[:,2], c='r', marker='o')

        matrixNInfarto = np.array(testInputs[np.where(predicts == 0)[0],:])
        ax.scatter(matrixNInfarto[:,0],matrixNInfarto[:,1] , matrixNInfarto[:,2], c='b', marker='o')


        plt.show()

    
    


    