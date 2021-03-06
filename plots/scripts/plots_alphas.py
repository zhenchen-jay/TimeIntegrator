import numpy as np
import matplotlib.pyplot as plt
import os

filePathPrefix = '/home/zchen96/Projects/TimeIntegrator/build_clion/output/Harmonic1d/'
timeStepTypes = ['0.005000']
timeStepLabels = ['5e-3']
stiffnessTypes = ['1.000000']
stiffnessLabels = ['1']
# materialTypes = ['Linear', 'NeoHookean']
materialTypes = ['Linear']
integratorTypes = ['Newmark', 'Implicit Euler', 'TRBDF2', 'BDF2']
fileName = '/alpha_'
theoFileName = '/alpha_theo_'
lambdaList = ['4400.04','4109.04','3652.85','3072.02', '2418.14', '1749.33', '1125','600.633','222.82','25.1306']

alphaList = []
alphaTheoList = []
time = []
for it in range(len(integratorTypes)):
    integrator = integratorTypes[it]
    alphaList.append([])
    alphaTheoList.append([])
    for n in range(10):
        filePath = filePathPrefix + timeStepTypes[0] + '_' + stiffnessTypes[0] + '_numSeg_10' + '/ConstantYoungs/' + materialTypes[0] + '/' + integrator + fileName + str(n) + '.txt'
        print(filePath)
        alphaData = np.loadtxt(filePath)
        filePath = filePathPrefix + timeStepTypes[0] + '_' + stiffnessTypes[0] + '_numSeg_10' + '/ConstantYoungs/' + materialTypes[0] + '/' + integrator + theoFileName+ str(n) + '.txt'
        alphaList[it].append(alphaData)
        alphaData = np.loadtxt(filePath)
        alphaTheoList[it].append(alphaData)

for i in range(len(alphaList[0][0])):
    time.append(i * 5e-3)


for i in range(len(integratorTypes)):
    for j in range(len(lambdaList)):
        eval = lambdaList[j]
        folderPath = filePathPrefix + timeStepTypes[0] + '_' + stiffnessTypes[0] + '_numSeg_10' + '/ConstantYoungs/'+ materialTypes[0] + '/SpectraAnalysis/'
        checkFolder = os.path.isdir(folderPath)
        if not checkFolder:
            os.makedirs(folderPath)
        plt.figure(dpi=300)
        plt.xlabel('t/s')
        plt.ylabel('alpha (consant Youngs and ' + materialTypes[0]+' material')
        startId = 3000
        endID = 3100
        plt.plot(time[startId:endID], alphaList[i][j][startId:endID], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[0] +'_' + eval)
        plt.plot(time[startId:endID], alphaTheoList[i][j][startId:endID], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[0] +'_' + eval + '_theoretical')
        plt.legend()
        imgPath = folderPath + integratorTypes[i] + '_' + eval + '_alphaDiff_zoomIn' + '.png'
        plt.savefig(imgPath)

        plt.figure(dpi=300)
        plt.xlabel('t/s')
        plt.ylabel('alpha (consant Youngs and ' + materialTypes[0]+' material')
        plt.plot(time, alphaList[i][j], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[0] +'_' + eval)
        plt.plot(time, alphaTheoList[i][j], linewidth=0.5, label = integratorTypes[i] + '-h=' + timeStepLabels[0] +'_' + eval + '_theoretical')
        plt.legend()
        imgPath = folderPath + integratorTypes[i] + '_' + eval + '_alphaDiff' + '.png'
        plt.savefig(imgPath)