
from Dataset import lerCSV,lerXLSX
import warnings
import sys 
import numpy as np
import pickle

from sklearn.cluster import KMeans

if __name__ == "__main__":
   
    with open("clf.pkl",'rb') as f:
        
        if len(sys.argv) > 1:
            path = sys.argv[1]
        else:
            path = "Banco de Dados - Infarto.xlsx"
        
        warnings.filterwarnings("ignore")
        
        if np.logical_or(".xlsx" in path,".XLSX" in path):  
            matrixTest = lerXLSX(path)
        else:
            matrixTest = lerCSV(path)
        
        clf = pickle.load(f)        
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
    