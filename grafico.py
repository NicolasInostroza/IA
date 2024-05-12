import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import animation

#------------------------------ 2.1 Trazar Datos ---------------------------------------#

x,y = np.loadtxt('data1.txt', usecols=[0,1], unpack=True) #Asignacion de valores a X e Y desde datos1.txt

plt.title("Gráfico de dispersión")
plt.xlabel('Poblacion de la ciudad en 10.000s')
plt.ylabel('Beneficio en $10.000s')
plt.xlim(2, 24)
plt.ylim(-4, 25)
plt.scatter(x, y, marker='x', color='red', label='Datos de entrenamiento')
plt.grid(True)
plt.legend()
plt.show()

#------------------------------ 2.2 Gradiente Descendente ------------------------------#

# Lo primero es setear los valores de theta0 y theta1 = 0 y la tasa de aprendizaje alpha = 0.01

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
    J_historial = np.zeros(num_iters)
    historial_theta = []
    divergencia_historia = []
    
    plt.ion()  #Modo interactivo que permite la actualizacion en tiempo real de los datos
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax.set_title("Gráfico de dispersión")
    ax.set_xlabel("Población de la ciudad en 10.000s")
    ax.set_ylabel("Beneficios en $10.000s")

    for iter in range(num_iters):
        theta0 = theta[0] - alpha * (1/m) * np.sum((x.dot(theta) - y) * x[:, 0])
        theta1 = theta[1] - alpha * (1/m) * np.sum((x.dot(theta) - y) * x[:, 1])
        
        theta = np.array([theta0, theta1])
        historial_theta.append(theta.copy())
        
        J_historial[iter] = funcioncosto(x, y, theta)

        ax.set_xlim(2, 24)
        ax.set_ylim(-4,25)
        ax2.set_ylim(0,10)

        if(iter % 10 == 0):

            #Regresión Lineal
            ax.set_title("Gráfico de dispersión")
            ax.set_xlabel("Población de la ciudad en 10.000s")
            ax.set_ylabel("Beneficios en $10.000s")
            ax.clear()
            ax.scatter(x[:, 1], y, marker='x', color='red', label='Datos de entrenamiento')
            ax.plot(x[:, 1], x.dot(theta), color='darkcyan', label='Regresion ajustandose')
            ax.grid(True)
            ax.legend()

            #divergencia

            divergencia_historia.append(J_historial[iter])
            ax2.set_xlabel('Iteraciones')
            ax2.set_ylabel('Costo J')
            ax2.set_title('Funcion de costo')
            ax2.clear()
            ax2.plot(range(len(divergencia_historia)), divergencia_historia, color='b')
            ax2.grid(True)

            plt.pause(0.00001)

    plt.ioff()
    plt.show()
    return theta, J_historial, historial_theta

x = np.column_stack((np.ones(len(y)), x)) #Se agrega una columna de unos al comienzo de la matriz X para tener en cuenta el sesgo
theta = np.zeros(x.shape[1]) #Crea un vector de coeficientes inicializado en ceros

theta_final, J_historial, historial_theta= GradienteDescendente(x, y, theta, alpha, num_iters)

print("\n##################################################################################################################") 
print("\nValor final de theta0 y theta1:", theta_final, "con alfa =",alpha, "Y número total de iteraciones =",num_iters)

#----------------------------------- Gráfico de gradiente descendente con los valores obtenidos de theta ------------------------------------------#

#plt.scatter(matriz_extendida[:, 1], y, marker='x', color='red')
#plt.xlim(2, 24)
#plt.ylim(-4, 25)
#plt.grid(True)
#plt.title("Gráfico de Regresión Lineal Obtenida")
#plt.xlabel("Población de la ciudad en 10.000s")
#plt.ylabel("Beneficios en $10.000s")

#slope, intercept, r, p, std_err = stats.linregress(x, y) #slope: Representa la pendiente de la línea de regresión lineal ajustada a los datos, 
                                                         #intercept: Representa la intersección en el eje y de la línea de regresión lineal.
                                                         #r: Representa el coeficiente de correlación, que indica la fuerza y la dirección de la relación lineal 
                                                         #p: Es el valor p asociado con el test de hipótesis de que la pendiente es igual a cero (no hay relación lineal).
                                                         #std_err: Es el error estándar de la pendiente estimada. Indica cuánto varía la pendiente estimada si se repitiera el experimento varias veces.

#def regresionL(x):
#  return slope * x + intercept

#regresionlineal = list(map(regresionL, x))

#plt.plot(matriz_extendida[:, 1], matriz_extendida.dot(theta_final), color='blue', label='Regresión lineal obtenida')
#plt.plot(x, regresionlineal, color='darkgray', label = 'Regresión lineal esperada')
#plt.legend()
#plt.show()

#---------------------------------- Calcular beneficios con los thetas obtenidos y en base a 70k y 35k de personas -----------------------------------#

personas = [3.5,7]

beneficio_70k = round(theta_final[0] + theta_final[1] * personas[1], 2)
beneficio_35k = round(theta_final[0] + theta_final[1] * personas[0], 2)

print("\nEl beneficio para 70.000 habitantes es de:", beneficio_70k,"\nEl beneficio para 35.000 habitantes es de:", beneficio_35k)

fig, ax = plt.subplots()
plt.scatter(x[:, 1], y, marker='x', color='red', label='Datos de entrenamiento')
plt.xlim(2, 24)
ax.set_ylim(-4, 25)
plt.scatter([3.5,7], [beneficio_35k,beneficio_70k], marker='x', color='green', label='Predicción de valores')
plt.plot([3.5,7], [beneficio_35k,beneficio_70k], color='blue')
ax.axvline(x = personas[0], ymin = 0, ymax = (beneficio_35k + 3)/25, color = 'green', linestyle = '--')
ax.axvline(x = personas[1], ymin = 0, ymax = (beneficio_70k + 3)/25, color = 'green', linestyle = '--')
ax.axhline(y = beneficio_35k, xmin = 0, xmax = (personas[0] - 2)/24, color = 'green', linestyle = '--')
ax.axhline(y = beneficio_70k, xmin = 0, xmax = (personas[1] - 2)/24, color = 'green', linestyle = '--')
label = f'({personas[0]}, {beneficio_35k})'
ax.text(personas[0], beneficio_35k, label, ha='right', va='bottom', color='black')
label = f'({personas[1]}, {beneficio_70k})'
ax.text(personas[1], beneficio_70k, label, ha='right', va='bottom', color='black')
plt.grid(True)
plt.title("Gráfico de Dispersión donde se indica el beneficio para 35.000 y 70.000 habitantes")
plt.xlabel("Población de la ciudad en 10.000s")
plt.ylabel("Beneficios en $10.000s")
plt.plot(x[:, 1], x.dot(theta_final), color='blue', label='Regresión lineal obtenida')
plt.legend()
plt.show()

#----------------------------------------------------------------- Gráfico 3D -----------------------------------------------------------------------------------#

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

# Calcula la función de costo J para cada combinación de theta0 y theta1
J_vals = np.zeros_like(theta0_grid)
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta = np.array([theta0_grid[i, j], theta1_grid[i, j]])
        J_vals[i, j] = funcioncosto(x, y, theta)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(theta0_grid, theta1_grid, J_vals, cmap='viridis')
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Función de Costo J')
ax.set_title('Función de Costo en 3D')

trayectoria, = ax.plot([], [], [], c='red', marker='o')

def update_3d(frame):
    theta_array = np.array(historial_theta)
    trayectoria.set_data(theta_array[:frame+1, 0], theta_array[:frame+1, 1])
    trayectoria.set_3d_properties(J_historial[:frame+1])
    return trayectoria,

# Se crea la animación utilizando la función FuncAnimation de matplotlib.
animacion_3d = FuncAnimation(fig, update_3d, frames=num_iters, interval=10, blit=True, repeat=False)

#Mostrar grafico
plt.show()

#-------------------------------------------------------------------- Funcion de costo bidimensional -----------------------------------------------------------#

# También puedes graficar la función de costo en un mapa de contorno

fig, ax = plt.subplots()
contour = plt.contour(theta0_grid, theta1_grid, J_vals, levels=np.logspace(-1, 3, 20))
plt.clabel(contour, inline=1, fontsize=10)
plt.xlabel('Theta0')
plt.ylabel('Theta1')
plt.title('Función de Costo - Contorno 2D')

trayectoria_2d, = ax.plot([], [], c='red', marker='o')

def update_2d(frame):
    theta_array = np.array(historial_theta)
    trayectoria_2d.set_data(theta_array[:frame+1, 0], theta_array[:frame+1, 1])#Se actualiza la información de los datos
    return trayectoria_2d,

animacion_2d = FuncAnimation(fig, update_2d, frames=num_iters, interval=10, blit=True, repeat=False)

plt.show()

#--------------------------------------------------------------------- Ecuación Normal ---------------------------------------------------------------------------#

theta_normal = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

print("\n##################################################################################################################") 
print("\nTheta calculado por la ecuación normal:", theta_normal)

plt.scatter(x[:, 1], y, marker='x', color='red')
plt.xlim(2, 24)
plt.ylim(-4, 25)
plt.grid(True)
plt.title("Gráfico de Regresión Lineal Obtenida")
plt.xlabel("Población de la ciudad en 10.000s")
plt.ylabel("Beneficios en $10.000s")

plt.plot(x[:, 1], x.dot(theta_final), color='blue', label='Regresión lineal obtenida mediante gradiente descendente')
plt.plot(x[:, 1], x.dot(theta_normal), color='darkgray', label = 'Regresión lineal obtenida mediante ecuación normal')
plt.legend()
plt.show()