import os.path
import csv
import numpy

def lerDataset():
    datasets_root_path = os.path.dirname(os.path.realpath(__file__))
    csvfile =  open(datasets_root_path+'/Banco de Dados - Infarto treinoTeste.csv', 'rb')
    return numpy.array(numpy.loadtxt(csvfile,delimiter=",",skiprows=1))
    
if __name__ == "__main__":
    print(lerDataset())