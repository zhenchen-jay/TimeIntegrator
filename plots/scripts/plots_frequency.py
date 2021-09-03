import numpy as np
import matplotlib.pyplot as plt

filePathPrefix = 'C:/Users/csyzz/Projects/TimeIntegrator/build/example/output/Harmonic1d/'
timeStepTypes = ['0.000050']
timeStepLabels = ['5e-5']
stiffnessTypes = ['0.000000']
stiffnessLabels = ['0']
integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
plotsType = ['peroid']
fileName = '/simulation_period.txt'

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

print("theoretic value is: ", 2 * np.pi * np.sqrt(0.15) / (10 * np.sqrt(10)))
for i in range(len(integratorTypes)):
    for j in range(len(plotsType)):
        plt.figure(dpi=300)
        plt.xlabel('t/s')
        plt.ylabel(plotsType[j])
        theoPeroid = np.full((len(integratorDataset[m][n][i][:, 0])), 2 * np.pi * np.sqrt(0.15) / (10 * np.sqrt(10)))
        for m in range(len(timeStepTypes)):
            for n in range(len(stiffnessTypes)):
                plotSeconds = len(integratorDataset[m][n][i][:, 0])
                plt.plot(integratorDataset[m][n][i][0:plotSeconds, 0], integratorDataset[m][n][i][0:plotSeconds, 1 + j], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[m])
                # plt.plot(integratorDataset[m][n][i][0:plotSeconds, 0], theoPeroid[0:plotSeconds], linewidth=0.5, label = 'theoretical')
        plt.legend()
        imgPath = filePathPrefix + 'period/' + integratorTypes[i] + '_' + plotsType[j] + '.png'
        plt.savefig(imgPath)

print("theoretic value is: ", 2 * np.pi * np.sqrt(0.15) / (10 * np.sqrt(10)))
for i in range(len(integratorTypes)):
    for j in range(len(plotsType)):
        plt.figure(dpi=300)
        plt.xlabel('t/s')
        plt.ylabel("error")
        theoPeroid = np.full((len(integratorDataset[m][n][i][:, 0])), 2 * np.pi * np.sqrt(0.15) / (10 * np.sqrt(10)))
        for m in range(len(timeStepTypes)):
            for n in range(len(stiffnessTypes)):
                plotSeconds = len(integratorDataset[m][n][i][:, 0])
                plt.plot(integratorDataset[m][n][i][0:plotSeconds, 0], np.divide(integratorDataset[m][n][i][0:plotSeconds, 1 + j] - theoPeroid[0:plotSeconds], theoPeroid[0:plotSeconds]), linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[m])
                # plt.plot(integratorDataset[m][n][i][0:plotSeconds, 0], theoPeroid[0:plotSeconds], linewidth=0.5, label = 'theoretical')
        plt.legend()
        imgPath = filePathPrefix + 'period_error/' + integratorTypes[i] + '_' + plotsType[j] + '.png'
        plt.savefig(imgPath)