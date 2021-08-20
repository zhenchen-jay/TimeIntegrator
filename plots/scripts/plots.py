import numpy as np
import matplotlib.pyplot as plt

filePathPrefix = 'E:/TimeIntegrator/output/Pogo_Stick/'
timeStepTypes = ['0.005000', '0.000500', '0.000050']
timeStepLabels = ['5e-3', '5e-4', '5e-5']
stiffnessTypes = ['10000000000.000000', '10000.000000']
stiffnessLabels = ['1e10', '1e4']
integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
plotsType = ['spring potential', 'gravity potential', 'IPC potential', 'kinetic energy', 'total energy', 'center of mass']
fileName = '/simulation_status.txt'

integratorDataset = []

for m in range(len(timeStepTypes)):
    integratorDataset.append([])
    for n in range(len(stiffnessTypes)):
        integratorDataset[m].append([])
        for integrator in integratorTypes:
            filePath = filePathPrefix + timeStepTypes[m] + '_' + stiffnessTypes[n] + '/' + integrator + fileName
            integratorData = np.loadtxt(filePath)
            integratorDataset[m][n].append(integratorData)

for m in range(len(timeStepTypes)):
    for n in range(len(stiffnessTypes)):
        for i in range(len(integratorTypes)):
            for j in range(len(plotsType)):
                plt.figure(dpi=300)
                plt.plot(integratorDataset[m][n][i][:, 0], integratorDataset[m][n][i][:, 1 + j], linewidth=0.5, label = integratorTypes[i])
                plt.xlabel('t/s')
                plt.ylabel(plotsType[j])
                plt.legend()
                imgPath = filePathPrefix + timeStepTypes[m] + '_' + stiffnessTypes[n] + '/' + integratorTypes[i] + '/' + plotsType[j] + '.png'
                plt.savefig(imgPath)

for i in range(len(integratorTypes)):
    for j in range(len(plotsType)):
        plt.figure(dpi=300)
        plt.xlabel('t/s')
        plt.ylabel(plotsType[j])
        for m in range(len(timeStepTypes)):
            for n in range(len(stiffnessTypes)):
                plt.plot(integratorDataset[m][n][i][:, 0], integratorDataset[m][n][i][:, 1 + j], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[m] + '-IPC_K=' + stiffnessLabels[n])
        plt.legend()
        imgPath = filePathPrefix + integratorTypes[i] + '_' + plotsType[j] + '.png'
        plt.savefig(imgPath)