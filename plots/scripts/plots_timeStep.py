import numpy as np
import matplotlib.pyplot as plt

filePathPrefix = 'C:/Users/csyzz/Projects/TimeIntegrator/build/output/Harmonic1d/'
timeStepTypes = ['0.005000']
timeStepLabels = ['5e-3']
stiffnessTypes = ['1.000000']
stiffnessLabels = ['1']
materialTypes = ['Linear', 'NeoHookean']
integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
plotsType = ['spring potential', 'gravity potential', 'IPC potential', 'kinetic energy', 'internal barrier', 'total energy', 'bottom displacement']
fileName = '/simulation_status.txt'

integratorDataset = []

for mt in range(len(materialTypes)):
    integratorDataset.append([])
    for m in range(len(timeStepTypes)):
        integratorDataset[mt].append([])
        for n in range(len(stiffnessTypes)):
            integratorDataset[mt][m].append([])
            for integrator in integratorTypes:
                filePath = filePathPrefix + timeStepTypes[m] + '_' + stiffnessTypes[n] + '_numSeg_10' + '/' + materialTypes[mt] + '/' + integrator + fileName
                print(filePath)
                integratorData = np.loadtxt(filePath)
                integratorDataset[mt][m][n].append(integratorData)

for mt in range(len(materialTypes)):
    for i in range(len(integratorTypes)):
        for j in range(len(plotsType)):
            if j <= 5:
                continue
            plt.figure(dpi=300)
            plt.xlabel('t/s')
            plt.ylabel(plotsType[j])
            for m in range(len(timeStepTypes)):
                for n in range(len(stiffnessTypes)):
                    plotSeconds = int(len(integratorDataset[mt][m][n][i][:, 0]))
                    plt.plot(integratorDataset[mt][m][n][i][0:plotSeconds, 0], integratorDataset[mt][m][n][i][0:plotSeconds, 1 + j], linewidth=0.5, label = materialTypes[mt] + '_' + integratorTypes[i] + '-h=' + timeStepLabels[m] + '-IPC_K=' + stiffnessLabels[n])
                    plt.legend()
                    imgPath = filePathPrefix + timeStepTypes[m] + '_' + stiffnessTypes[n] + '_numSeg_10' + '/bottomDis/' + materialTypes[mt] + '_' + integratorTypes[i] + '_' + plotsType[j] + '.png'
                    plt.savefig(imgPath)