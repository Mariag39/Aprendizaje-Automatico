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

def coste(thetha,x,y):
    h = np.dot(x,thetha)
    aux = (h - y) ** 2
    return aux.sum()/(2.0*float(len(x)))


def draw(thetha):
    xx = np.linspace(min(x[:,1]),max(y))
    yy = h(xx,thetha)

    plt.plot(xx,yy,'-')
def gradient(thethi,alpha, muestra,thetha,x):
    suma = 0.0

    for i in range(muestra):
        if(thethi == thetha[0]):
            suma = suma + (h(x[i],thetha) - y[i])
        else:
            suma = suma + ((h(x[i],thetha) - y[i]) * x[i])
    
    return thethi - ((alpha/muestra)*suma)


def des_gradient(x,y, alpha):
    thetha = np.arange(2)
    
    coste_min = coste(thetha,x,y)

    for j in range(0,1500):
        suma1 = gradient(thetha[0],alpha,thetha,x)
        suma2 = gradient(thetha[1],alpha,thetha,x)
        thetha[0] = suma1
        thetha[1] = suma2
        coste_tmp = coste(thetha,x,y)
        
        if(coste_min > coste_tmp):
            taux0 = thetha[0]
            taux1 = thetha[1]
            coste_min = coste_tmp
        
    draw(thetha)
    return taux0,taux1
    


thetha = [0,0]
thetha[0] = 0
thetha[1] = 0
alpha = 0.01
data = carga_csv("ex1data1.csv")
x = data[:, :-1]
np.shape(x)
y = data[:, -1]
np.shape(y)

m = np.shape(x)[0]
n = np.shape(x)[1]

x = np.hstack([np.ones([m,1]),x])
    
thetha,costes = des_gradient(x,y, alpha) #se tuerce ARREGLAR




plt.scatter(x[:,1],y[:,0],color="red", marker='x')
plt.xlabel('Población de la ciudad en 10,000s')
plt.ylabel('Ingresos en $10,000s')

plt.show()



#calcular h 
#columnas de 1 a columna de x. sirve para qie el calculo de x se haga cp,p theta * x. M filas que tiene la columna 
#calculo de j para saver que todo va bien, no es imprescindible pero sirve de depuraciónimport numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as integrate
import random as rnd
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    valores = read_csv(file_name, header=None).values

    return valores.astype(float)


def h(x,thetha):
    return thetha[0] + x*thetha[1]

def coste(thetha,x,y):
    h = np.dot(x,thetha)
    aux = (h - y) ** 2
    return aux.sum()/(2.0*float(len(x)))


def draw(thetha):
    xx = np.linspace(min(x[:,1]),max(y))
    yy = h(xx,thetha)

    plt.plot(xx,yy,'-')
def gradient(thethi,alpha, muestra,thetha,x):
    suma = 0.0

    for i in range(muestra):
        if(thethi == thetha[0]):
            suma = suma + (h(x[i],thetha) - y[i])
        else:
            suma = suma + ((h(x[i],thetha) - y[i]) * x[i])
    
    return thethi - ((alpha/muestra)*suma)


def des_gradient(x,y, alpha):
    thetha = np.arange(2)
    
    coste_min = coste(thetha,x,y)

    for j in range(0,1500):
        suma1 = gradient(thetha[0],alpha,thetha,x)
        suma2 = gradient(thetha[1],alpha,thetha,x)
        thetha[0] = suma1
        thetha[1] = suma2
        coste_tmp = coste(thetha,x,y)
        
        if(coste_min > coste_tmp):
            taux0 = thetha[0]
            taux1 = thetha[1]
            coste_min = coste_tmp
        
    draw(thetha)
    return taux0,taux1
    


thetha = [0,0]
thetha[0] = 0
thetha[1] = 0
alpha = 0.01
data = carga_csv("ex1data1.csv")
x = data[:, :-1]
np.shape(x)
y = data[:, -1]
np.shape(y)

m = np.shape(x)[0]
n = np.shape(x)[1]

x = np.hstack([np.ones([m,1]),x])
    
thetha,costes = des_gradient(x,y, alpha) #se tuerce ARREGLAR




plt.scatter(x[:,1],y[:,0],color="red", marker='x')
plt.xlabel('Población de la ciudad en 10,000s')
plt.ylabel('Ingresos en $10,000s')

plt.show()



#calcular h 
#columnas de 1 a columna de x. sirve para qie el calculo de x se haga cp,p theta * x. M filas que tiene la columna 
#calculo de j para saver que todo va bien, no es imprescindible pero sirve de depuración
