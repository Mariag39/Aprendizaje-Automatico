import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
import random as rnd
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values

    return valores.astype(float)



thetha = [0,0]
thetha[0] = 0
thetha[1] = 0
alpha = 0.1
def h(x,thetha):
    return thetha[0] + x*thetha[1]

x = []
x = carga_csv("ex1data1.csv")


def gradiente(x):
    content = [0]
    xc = []
    holi = []
    for row in x:
        xc = np.array([row[i]for i in content])
       
        holi = np.hstack((1,xc))
        print(holi)   
  
   
    
   
    


print(gradiente(x))   

#calcular h 
#columnas de 1 a columna de x. sirve para qie el calculo de x se haga cp,p theta * x. M filas que tiene la columna 
#calculo de j para saver que todo va bien, no es imprescindible pero sirve de depuración
