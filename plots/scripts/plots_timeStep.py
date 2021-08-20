import numpy as np
import matplotlib.pyplot as plt

filePathPrefix = 'E:/TimeIntegrator/output/Pogo_Stick/'
timeStepTypes = ['0.005000', '0.000500', '0.000050', '0.000005']
timeStepLabels = ['5e-3', '5e-4', '5e-5', '5e-6']
stiffnessTypes = ['10000.000000']
stiffnessLabels = ['1e4']
# integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
integratorTypes = ['TRBDF2']
plotsType = ['spring potential', 'gravity potential', 'IPC potential', 'kinetic energy', 'total energy', 'center of mass']
fileName = '/simulation_status.txt'

integratorDataset = []

for m in range(len(timeStepTypes)):
    integratorDataset.append([])
    for n in range(len(stiffnessTypes)):
        integratorDataset[m].append([])
        for integrator in integratorTypes:
            filePath = filePathPrefix + timeStepTypes[m] + '_' + stiffnessTypes[n] + '/' + integrator + fileName
            print(filePath)
            integratorData = np.loadtxt(filePath)
            integratorDataset[m][n].append(integratorData)

for i in range(len(integratorTypes)):
    for j in range(len(plotsType)):
        plt.figure(dpi=300)
        plt.xlabel('t/s')
        plt.ylabel(plotsType[j])
        for m in range(len(timeStepTypes)):
            for n in range(len(stiffnessTypes)):
                plotSeconds = int(len(integratorDataset[m][n][i][:, 0]) / 2)
                if timeStepTypes[m] == '0.000005':
                    plotSeconds = len(integratorDataset[m][n][i][:, 0])
                plt.plot(integratorDataset[m][n][i][0:plotSeconds, 0], integratorDataset[m][n][i][0:plotSeconds, 1 + j], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[m] + '-IPC_K=' + stiffnessLabels[n])
        plt.legend()
        imgPath = filePathPrefix + 'timeStep/' + integratorTypes[i] + '_' + plotsType[j] + '.png'
        plt.savefig(imgPath)