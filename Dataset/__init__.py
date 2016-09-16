import os.path
import csv
import numpy as np
from openpyxl import load_workbook


def lerDataset():
    datasets_root_path = os.path.dirname(os.path.realpath(__file__))
    csvfile =  open(datasets_root_path+'/Banco de Dados - Infarto treinoTeste.csv', 'rb')
    return np.array(numpy.loadtxt(csvfile,delimiter=",",skiprows=1))


def lerXLSX(path, fullPath = False,sheetName="Hoja1"):
    if(fullPath):
       datasets_root_path =  "" + path
    else:
        datasets_root_path = os.path.dirname(os.path.realpath(__file__)) +"/" +path

    try:
        wb = load_workbook(filename = datasets_root_path)
    except ValueError:
        print("Erro ao carregar arquivo .xlsx\n"+ValueError)
        raise
    
    try:
        ws = wb['Hoja1']
    except ValueError:
        print("Erro ao recuperar planilha. Planilha de nome "+sheetName+" nao encontrada.")
        raise
    
    try:
        matrix = np.array([[ float(cell.value) for cell in row] for row in ws.rows[1:]])
    except ValueError:
        print("Erro ao recuperar planilha. Planilha de nome "+sheetName+" nao encontrada.")
        raise
    
    return matrix

    
if __name__ == "__main__":
    print(lerDataset())