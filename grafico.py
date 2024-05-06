import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.animation import FuncAnimation
import time

#------------------------------ 2.1 Trazar Datos ---------------------------------------#

x,y = np.loadtxt('data1.txt', usecols=[0,1], unpack=True) #Asignacion de valores a X e Y desde datos1.txt

plt.title("Gráfico de dispersión")
plt.xlabel('Poblacion de la ciudad en 10.000s')
plt.ylabel('Beneficio en $10.000s')
plt.scatter(x, y, marker='x', color='red', label='Datos de entrenamiento')
plt.grid(True)
plt.legend()
plt.show()

#------------------------------ 2.2 Gradiente Descendente ------------------------------#

# Lo primero es setear los valores de theta0 y theta1 = 0 y la tasa de aprendizaje alpha = 0.01

theta = 0
theta0 = 0
theta1 = 0
alpha = 0.01
num_iters = 1500

# Lo siguiente es calcular la función de costo

def funcioncosto(x, y, theta):
    m = len(y)
    J = (1/(2*m)) * np.sum((x.dot(theta) - y)**2) #np.sum permite realizar una sumatoria y pertenece a la libreria numpy
    return J

print ("El costo es de:", funcioncosto(x, y, 0)) 

# Después definimos la gradiente descendente

def GradienteDescendente(x, y, theta, alpha, num_iters):
    m = len(y)
    J_historia = np.zeros(num_iters)
    historial_theta = []
    
    plt.ion()  #Modo interactivo que permite la actualizacion en tiempo real de los datos
    fig, ax = plt.subplots()
    
    for iter in range(num_iters):
        theta0 = theta[0] - alpha * (1/m) * np.sum((x.dot(theta) - y) * x[:, 0])
        theta1 = theta[1] - alpha * (1/m) * np.sum((x.dot(theta) - y) * x[:, 1])
        
        theta = np.array([theta0, theta1])
        historial_theta.append(theta.copy())
       
        
        J_historia[iter] = funcioncosto(x, y, theta)

        ax.clear()
        ax.scatter(x[:, 1], y, marker='x', color='red', label='Datos de entrenamiento')
        ax.plot(x[:, 1], x.dot(theta), color='darkcyan', label='Regresion ajustandose')
        ax.grid(True)
        ax.set_title("Gráfico de dispersión")
        ax.set_xlabel("Población de la ciudad en 10.000s")
        ax.set_ylabel("Beneficios en $10.000s")
        ax.legend()
        plt.pause(0.0001)
        print("\nValores de theta0 y theta1 actualmente son:", theta)

    plt.ioff()
    return theta, J_historia, historial_theta

matriz_extendida = np.column_stack((np.ones(len(y)), x)) #Se agrega una columna de unos al comienzo de la matriz X para tener en cuenta el sesgo
theta = np.zeros(matriz_extendida.shape[1])

theta_final, J_historia, historial_theta = GradienteDescendente(matriz_extendida, y, theta, alpha, num_iters)

print("\n##################################################################################################################") 
print("\nValor final de theta0 y theta1:", theta_final, "con alfa =",alpha, "Y número total de iteraciones =",num_iters)

#----------------------------------- Gráfico de gradiente descendente con los valores obtenidos de theta ------------------------------------------#

plt.scatter(matriz_extendida[:, 1], y, marker='x', color='red')
plt.xlim(4, 24)
plt.grid(True)
plt.title("Gráfico de Regresión Lineal Obtenida")
plt.xlabel("Población de la ciudad en 10.000s")
plt.ylabel("Beneficios en $10.000s")

slope, intercept, r, p, std_err = stats.linregress(x, y) #slope: Representa la pendiente de la línea de regresión lineal ajustada a los datos, 
                                                         #intercept: Representa la intersección en el eje y de la línea de regresión lineal.
                                                         #r: Representa el coeficiente de correlación, que indica la fuerza y la dirección de la relación lineal 
                                                         #p: Es el valor p asociado con el test de hipótesis de que la pendiente es igual a cero (no hay relación lineal).
                                                         #std_err: Es el error estándar de la pendiente estimada. Indica cuánto varía la pendiente estimada si se repitiera el experimento varias veces.

def regresionL(x):
  return slope * x + intercept

mymodel = list(map(regresionL, x))

plt.plot(matriz_extendida[:, 1], matriz_extendida.dot(theta_final), color='blue', label='Regresión lineal obtenida')
plt.plot(x, mymodel, color='darkgray', label = 'Regresión lineal esperada')
plt.legend()
plt.show()

#---------------------------------- Calcular beneficios con los thetas obtenidos y en base a 70k y 35k de personas -----------------------------------#

beneficio_70k = theta_final[0] + theta_final[1] * 7
beneficio_35k = theta_final[0] + theta_final[1] * 3.5

print("\nEl beneficio para 70.000 habitantes es de:", beneficio_70k,"\nEl beneficio para 35.000 habitantes es de:", beneficio_35k)

plt.scatter(x, y, marker='x', color='red', label='Datos de entrenamiento')
plt.axhline(beneficio_70k, linestyle = 'dotted', marker='x', color='green')
plt.axvline(beneficio_35k, linestyle = 'dotted', marker='x', color='green')
plt.plot(beneficio_70k, beneficio_35k, color='green', marker='x')
plt.grid(True)
plt.title("Gráfico de Dispersión donde se indica el beneficio para 35.000 y 70.000 habitantes")
plt.xlabel("Población de la ciudad en 10.000s")
plt.ylabel("Beneficios en $10.000s")
plt.legend()
plt.show()