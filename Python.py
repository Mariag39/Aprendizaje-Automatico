import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
import random as rnd
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values

    return valores.astype(float)


def h(x,thetha):
    return thetha[0] + x*thetha[1]

def coste(thetha, size,x,y):
    suma = 0.0
    for i in range(0,size):
        suma = suma + ((h(x[i],thetha) - y[i])**2)
    return suma/(2.0*float(size))



def linear_regression(valores):
    arr = np.transpose(np.array(valores[:-1]))
    x = np.transpose(np.array(arr[:-1]))
    #[:-1] omite los valores de y
    y = np.transpose(np.array(arr[-1:]))
    size = y.size
    aux = np.ones((len(x),1))
    res = np.hstack((aux,x))
    return res,y,size

def scatter_data(x,y):
    plt.figure()
    plt.scatter(x[:,1],y[:,0],color="red", marker='x')
    plt.xlabel('Población de la ciudad en 10,000s')
    plt.ylabel('Ingresos en $10,000s')
    plt.show()

thetha = [0,0]
thetha[0] = 0
thetha[1] = 0
alpha = 0.1
data = []
data = carga_csv("ex1data1.csv")
x,y,size = linear_regression(data)
scatter_data(x,y)
print(x)
print(y)
cost = coste(thetha, size,x,y)
print(cost)
#calcular h 
#columnas de 1 a columna de x. sirve para qie el calculo de x se haga cp,p theta * x. M filas que tiene la columna 
#calculo de j para saver que todo va bien, no es imprescindible pero sirve de depuración
