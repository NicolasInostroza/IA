import matplotlib.pyplot as plt
import numpy as np

x,y = np.loadtxt('data1.txt', usecols=[0,1], unpack=True)
m,b = np.polyfit(x, y, deg=1)

def menu():
    print("##########################Bienvenido##########################")
    op = int(input("Ingrese el número de fila para el eje X: "))
    while(op == 0 or op > 96):
        op=int(input("Error, reingrese opción: "))

    ejex = x[op - 1]
    ejey = y[op - 1]
    print(ejex, ",", ejey)

    plt.title("Datos IA")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axhline(ejey, xmin = 0, xmax = 10, linestyle = 'dotted')
    plt.axvline(ejex, ymin = 0, ymax = 10, linestyle = 'dotted')
    plt.legend(bbox_to_anchor = (1.0, 1))
    plt.plot(x,y, "*")
    plt.plot(x, m*x + b)
    plt.grid(True)
    plt.show()

menu()