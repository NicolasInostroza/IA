import matplotlib.pyplot as plt
import numpy as np

x,y = np.loadtxt('data1.txt', usecols=[0,1], unpack=True)

def menu():
    print("##########################Bienvenido##########################")
    op = int(input("Ingrese el número de fila para el eje X: "))
    while(op == 0 or op > 97):
        op=int(input("Error, reingrese opción: "))

    vector1 = (x[op - 1], y[op - 1])
    print(vector1)

    op2 = int(input("Ingrese el número de fila para el eje X: "))
    while(op2 == 0 or op2 > 97):
        op=int(input("Error, reingrese opción: "))

    vector2 = (x[op2 - 1], y[op2 - 1])
    print(vector2)

    plt.title("Datos IA")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x,y, "*r")
    plt.plot(vector1, vector2, linewidth=2, color='b')
    plt.grid(True)

    plt.show()

menu()