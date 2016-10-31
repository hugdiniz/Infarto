import os.path
import csv
import numpy as np
from openpyxl import load_workbook
from sklearn.cluster import KMeans
import pickle
from sklearn import preprocessing

def probabilidadeInfarto(matrixTrain,training=True):
    
    if training:
        est = KMeans(n_clusters=8, n_init=1, init='random')
        
        with open("kmeansPI.pkl",'wb') as f:
            pickle.dump(est,f)  
    else:
        est = pickle.load(open("kmeansPI.pkl",'rb'))


    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-2, 2))
    matrixProcessed = min_max_scaler.fit_transform(matrixTrain[:,:3])    
    matrixProcessed = np.array([[matrixProcessed[y,0],np.tanh(matrixProcessed[y,1]),np.tanh(matrixProcessed[y,2])]  for y in range(0,len(matrixTrain[:,0]))])
    
    est.fit(matrixProcessed[:,:3])
    W = matrixTrain[:,3]
    T = np.array([ est.predict(matrixProcessed[x,:3])[0] for x in range(0,len(matrixProcessed))])
        
    class1 = np.where(T == 0 )[0]
    class2 = np.where(T == 1 )[0]
    class3 = np.where(T == 2 )[0]
    class4 = np.where(T == 3 )[0]
    class5 = np.where(T == 4 )[0]
    class6 = np.where(T == 5 )[0]
    class7 = np.where(T == 6 )[0]
    class8 = np.where(T == 7 )[0]      
          
    matrixTrain = np.array([np.concatenate((matrixTrain[x,:3],[0,0,0,0,0,0,0,0],[matrixTrain[x,3]]),axis=0) for x in range(0,len(matrixTrain))])
    
    for x in range(0,len(matrixTrain)):        
        if x in class1:
           matrixTrain[x,3] = 1            
        if x in class2:
            matrixTrain[x,4] = 1            
        if x in class3:
            matrixTrain[x,5] = 1            
        if x in class4:
            matrixTrain[x,6] = 1
        if x in class5:
            matrixTrain[x,7] = 1
        if x in class6:
            matrixTrain[x,8] = 1
        if x in class7:
            matrixTrain[x,9] = 1
        if x in class8:
            matrixTrain[x,10] = 1            
    
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



def lerXLSX(path, fullPath = False,sheetName="Hoja1",limitCellPerRow = 4,training=False,agrupamento = True):
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
    
    if agrupamento:
        return probabilidadeInfarto(matrix,training)
    else:
        return matrix

    
if __name__ == "__main__":
    print(lerDataset())