import os.path
import csv
import numpy as np
from openpyxl import load_workbook
from sklearn.cluster import KMeans
import pickle

def probabilidadeInfarto(matrixTrain,training=True):
    
    if training:
        est = KMeans(n_clusters=4, n_init=1, init='random')
        est.fit(matrixTrain[:,:3])
        with open("kmeansPI.pkl",'wb') as f:
            pickle.dump(est,f)  
    else:
        est = pickle.load(open("kmeansPI.pkl",'rb'))
    
    W = matrixTrain[:,3]
    T = np.array([ est.predict(matrixTrain[x,:3])[0] for x in range(0,len(matrixTrain))])
        
    class1 = np.where(T == 0 )[0]
    class2 = np.where(T == 1 )[0]
    class3 = np.where(T == 2 )[0]
    class4 = np.where(T == 3 )[0]  
    
    class1T = np.where(W[class1] == 1 )[0]
    class2T = np.where(W[class2] == 1 )[0]
    class3T = np.where(W[class3] == 1 )[0]
    class4T = np.where(W[class4] == 1 )[0]    

    p1 = float(len(class1T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))    
    
    p2 = float(len(class2T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))  
    
    p3 = float(len(class3T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))   
    
    p4 = float(len(class4T)) / float(len(class1T) + len(class2T) + len(class3T) + len(class4T))    
       
    matrixTrain = np.array([np.concatenate((matrixTrain[x,:3],[0],[matrixTrain[x,3]]),axis=0) for x in range(0,len(matrixTrain))])
    
    for x in range(0,len(matrixTrain)):        
        if x in class1:
           matrixTrain[x,3] = p1            
        if x in class2:
            matrixTrain[x,3] = p2            
        if x in class3:
            matrixTrain[x,3] = p3            
        if x in class4:
            matrixTrain[x,3] = p4            
    
    return matrixTrain

def lerCSV(path, fullPath = False,limitCellPerRow = 4,training=False):
    if(fullPath):
       datasets_root_path =  "" + path
    else:
       datasets_root_path = os.path.dirname(os.path.realpath(__file__)) +"/" +path

    try:       
       csvfile =  open(datasets_root_path, 'rb')
    except ValueError:
        print("Erro ao carregar arquivo .xlsx\n"+ValueError)
        exit()
        
    try:
        matrix = np.array(np.loadtxt(csvfile,delimiter=",",skiprows=1))
    except ValueError:
        print("Erro ao recuperar planilha em CSV, erro na planilha, ela esta seguindo os padroes do csv especificado no projeto ?")
        raise
        exit()
    
    return probabilidadeInfarto(matrix,training)



def lerXLSX(path, fullPath = False,sheetName="Hoja1",limitCellPerRow = 4,training=False):
    if(fullPath):
       datasets_root_path =  "" + path
    else:
        datasets_root_path = os.path.dirname(os.path.realpath(__file__)) +"/" +path

    try:
        wb = load_workbook(filename = datasets_root_path)
    except ValueError:
        print("Erro ao carregar arquivo .xlsx\n"+ValueError)
        exit()
    
    try:
        ws = wb['Hoja1']
    except ValueError:
        print("Erro ao recuperar planilha. Planilha de nome "+sheetName+" nao encontrada.")
        exit()
    
    try:
        matrix = np.array([[ float(cell.value) for cell in row[:limitCellPerRow]] for row in ws.rows[1:]])
    except ValueError:
        print("Erro ao recuperar planilha. Planilha de nome "+sheetName+" nao encontrada.")
        exit()
    
    return probabilidadeInfarto(matrix,training)

    
if __name__ == "__main__":
    print(lerDataset())