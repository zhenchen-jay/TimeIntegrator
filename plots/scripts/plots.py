import numpy as np
import matplotlib.pyplot as plt

filePathPrefix = 'C:/Users/csyzz/Projects/TimeIntegrator/build/output/Pogo_Stick/'
integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
fileName = '/simulation_status.txt'

integratorDataset = []

for integrator in integratorTypes:
    filePath = filePathPrefix + integrator + fileName
    integratorData = np.loadtxt(filePath)
    integratorDataset.append(integratorData)

plt.figure(dpi=300)
for i in range(1, len(integratorTypes)):
    plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 1], linewidth=0.5, label = integratorTypes[i])
    plt.xlabel('t/s')
    plt.ylabel('spring potential')
plt.legend()
plt.show()

plt.figure(dpi=300)
for i in range(1, len(integratorTypes)):
    plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 2], linewidth=0.5, label = integratorTypes[i])
    plt.xlabel('t/s')
    plt.ylabel('spring potential')
plt.legend()
plt.show()

plt.figure(dpi=300)
for i in range(1, len(integratorTypes)):
    plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 3], linewidth=0.5, label = integratorTypes[i])
    plt.xlabel('t/s')
    plt.ylabel('spring potential')
plt.legend()
plt.show()

plt.figure(dpi=300)
for i in range(1, len(integratorTypes)):
    plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 4], linewidth=0.5, label = integratorTypes[i])
    plt.xlabel('t/s')
    plt.ylabel('spring potential')
plt.legend()
plt.show()

plt.figure(dpi=300)
for i in range(1, len(integratorTypes)):
    plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 5], linewidth=0.5, label = integratorTypes[i])
    plt.xlabel('t/s')
    plt.ylabel('spring potential')
plt.legend()
plt.show()

plt.figure(dpi=300)
for i in range(len(integratorTypes)):
    plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 6], linewidth=0.5, label = integratorTypes[i])
    plt.xlabel('t/s')
    plt.ylabel('spring potential')
plt.legend()
plt.show()