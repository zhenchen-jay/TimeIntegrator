import numpy as np
import matplotlib.pyplot as plt

filePathPrefix = '../build_clion/output/Pogo_Stick/'
integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
plotsType = ['spring potential', 'gravity potential', 'IPC potential', 'kinetic energy', 'total energy', 'center of mass']
fileName = '/simulation_status.txt'

integratorDataset = []

for integrator in integratorTypes:
    filePath = filePathPrefix + integrator + fileName
    integratorData = np.loadtxt(filePath)
    integratorDataset.append(integratorData)


for i in range(len(integratorTypes)):
    for j in range(len(plotsType)):
        plt.figure(dpi=300)
        plt.plot(integratorDataset[i][:, 0], integratorDataset[i][:, 1], linewidth=0.5, label = integratorTypes[i])
        plt.xlabel('t/s')
        plt.ylabel(plotsType[j])
        plt.legend()
        imgPath = filePathPrefix + integratorTypes[i] + '/' + plotsType[j] + '.png'
        plt.savefig(imgPath)