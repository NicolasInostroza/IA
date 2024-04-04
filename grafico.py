import matplotlib.pyplot as plt
import numpy as np

x,y = np.loadtxt('data1.txt', usecols=[0,1], unpack=True)

print(x,y)

plt.title("Datos IA")
plt.xlabel("X")
plt.ylabel("Y")

plt.plot(x,y, "*r")

plt.grid(True)

plt.show()