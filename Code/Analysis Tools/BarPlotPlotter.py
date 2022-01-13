import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import sys

from pathlib import Path
from PValueToLetterConverter import PValueToLetterConverter
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

class BarPlotPlotter (object):

    # plot bar plots of all bar result figures
    def __init__(self, baseResultsFolder, selectedFolders,
                 addOtherTestWithBaseFolder=None,
                 furtherFolder="svm_k2h_combinedTable_l3f0n1c0bal1ex0/",
                 randFilename="combinedResultsWithTestingOf_1000_randomizedRuns_ex1.csv",
                 resultFilename="results.csv", resultsTestFilename="resultsWithOnlyTesting.csv", performanceIdx=1,
                 plotOnlyRandom=False, filenameToSave="",
                 compareRandAndNorm=True, minY=0, doSpecial=False, nrOfReplicates=6):
        self.baseResultsFolder = baseResultsFolder
        self.selectedFolders = selectedFolders
        self.addOtherTestWithBaseFolder = addOtherTestWithBaseFolder
        self.furtherFolder = furtherFolder
        self.randFilename = randFilename
        self.resultFilename = resultFilename
        self.resultsTestFilename = resultsTestFilename
        self.performanceIdx = performanceIdx
        self.plotOnlyRandom = plotOnlyRandom
        self.compareRandAndNorm = compareRandAndNorm or doSpecial
        self.minY = minY
        self.doSpecial = doSpecial
        self.filenameToSave = filenameToSave
        self.nrOfReplicates = nrOfReplicates
        self.loadFiles()
        self.createFigures(self.performanceIdx, self.compareRandAndNorm,
                           minY=self.minY)

    def loadFiles(self):
        self.randTable = []
        self.testResultTables = None
        self.furtherTestResults = None
        self.resultsTable = self.loadTables(self.baseResultsFolder) # add check for number of replicates
        if (self.plotOnlyRandom or self.compareRandAndNorm) and not self.doSpecial:
            self.randTable = self.loadTables(self.baseResultsFolder, addFurtherFolder=False,
                                             addSpecificNameSuffix=self.randFilename)
        else:
            if not self.resultsTestFilename is None:
                self.testResultTables = self.loadTables(self.baseResultsFolder, addFurtherFolder=True,
                                                 addSpecificNameSuffix=self.resultsTestFilename)
        if not self.addOtherTestWithBaseFolder is None:
            self.furtherTestResults = self.loadTables(self.addOtherTestWithBaseFolder, addFurtherFolder=True,
                                             addSpecificNameSuffix=self.resultsTestFilename)

    def loadTables(self, baseFolder, addFurtherFolder=True, addSpecificNameSuffix=None):
        resultsTables = []
        for resultsFolder in self.selectedFolders:
            currentResultsFolder = baseFolder + resultsFolder + "/"
            resultsFilename = currentResultsFolder
            if addFurtherFolder:
                resultsFilename += self.furtherFolder
            if addSpecificNameSuffix:
                resultsFilename += addSpecificNameSuffix
            else:
                resultsFilename += self.resultFilename
            resultsTable = pd.read_csv(resultsFilename, index_col=0)
            resultsTables.append(resultsTable)
        return resultsTables

    def createFigures(self, performanceIdx=1, compareRandAndNorm=True, minY=0,
                      printPValues=True):
        x_pos, mean, std, colors = self.setupData(performanceIdx, compareRandAndNorm,
                                                  nrOfReplicates=self.nrOfReplicates,
                                                  useTesting=not self.testResultTables is None)
        statisticsLetters = ""
        statisticsLettersFilename = Path(self.filenameToSave).with_name(Path(self.filenameToSave).stem + "_statisticsLetters.txt")
        if not np.all(np.asarray(std) == 0):
            pValueTable = self.calcTrainAndValDifferences(performanceIdx, compareRandAndNorm,
                                               nrOfReplicates=self.nrOfReplicates,
                                               correctPValues=True, printPValues=printPValues)
            pValueTableName = Path(self.filenameToSave).with_name(Path(self.filenameToSave).stem + "_trainValPValues.csv")
            pValueTable.to_csv(pValueTableName)
            trainingGroupLetters = PValueToLetterConverter(pValueTable.rename(columns={"training p-values":"p-value"})).GetGroupNameLetters()
            valGroupLetters = PValueToLetterConverter(pValueTable.rename(columns={"validation p-values":"p-value"})).GetGroupNameLetters()
            statisticsLetters += f"trainingGroupLetters {trainingGroupLetters}\n"
            statisticsLetters += f"valGroupLetters {valGroupLetters}\n"
        with open(statisticsLettersFilename, "w") as file:
            file.write(statisticsLetters)
        yLabel = self.setYLabel(performanceIdx)
        # Build the plot
        plt.rcParams.update({'font.size': 18})
        if len(self.selectedFolders) != 3:
            xFigSize = 6.4 * len(self.selectedFolders) / 3
        else:
            xFigSize = 6.4
        fig, ax = plt.subplots(figsize=(xFigSize, 4.8))
        ax.bar(x_pos, mean, yerr=std, align='center', ecolor='black',
               color=colors, capsize=10, edgecolor ="black")
        if self.plotOnlyRandom:
            minY = 0
        if "%" in yLabel:
            maxY = 100
            yTicksAndLabels = np.arange(minY, 101, 10)
        else:
            maxY = 1
            yTicksAndLabels = np.arange(minY, 1.1, 0.1)
            yTicksAndLabels = np.round(yTicksAndLabels, 2)
        ax.set_yticks(yTicksAndLabels)
        ax.set_yticklabels(yTicksAndLabels)
        ax.set_ylim((minY, maxY))
        ax.set_ylabel(yLabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels("")
        # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
        ax.yaxis.grid(False)

        #remove  top and right box line
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().set_ticks([])

        # Save the figure and show
        plt.tight_layout()
        if self.filenameToSave:
            plt.savefig(self.filenameToSave)
            plt.close()
        else:
            plt.show()

    def setupData(self, performanceIdx, compareRandAndNorm, nrOfReplicates=5, useTesting=True):
        if self.doSpecial:
            mean = self.doSpecial["mean"]
            std = self.doSpecial["std"]
            coloryByHtml = self.doSpecial["coloryByHtml"]
            idx = self.doSpecial["idx"]
        else:
            if compareRandAndNorm:
                mean, std = self.extractMeanAndStd(performanceIdx, nrOfReplicates=nrOfReplicates)
                coloryByHtml = ["#00aeff", "#ffb100", "#004f99", "#ff6900"]
                idx = 4
            else:
                mean, std = self.extractMeanAndStd(performanceIdx, nrOfReplicates=nrOfReplicates,
                                            plotOnlyRandom=self.plotOnlyRandom)
                idx = 2
                coloryByHtml = ["#2E75B6", "#548235"]
                if useTesting:
                    mean, std = self.addTestData(mean, std, idx, performanceIdx, self.testResultTables)
                    idx += 1
                    coloryByHtml.append("#ff5800")
                if self.furtherTestResults:
                    mean, std = self.addTestData(mean, std, idx, performanceIdx, self.furtherTestResults)
                    idx += 1
                    coloryByHtml.append("#d1008f")
        x_pos = np.arange(len(mean))
        x_pos += np.arange(len(mean)) // idx
        colors = np.full(len(x_pos), coloryByHtml[0])
        colors[[i%idx==1 for i in range(len(mean))]] = coloryByHtml[1]
        if compareRandAndNorm:
            colors[[i%idx==2 for i in range(len(mean))]] = coloryByHtml[2]
            colors[[i%idx==3 for i in range(len(mean))]] = coloryByHtml[3]
        else:
            if useTesting:
                colors[[i%idx==2 for i in range(len(mean))]] = coloryByHtml[2]
            if self.furtherTestResults:
                colors[[i%idx==3 for i in range(len(mean))]] = coloryByHtml[2+useTesting]
        return x_pos, mean, std, colors

    def extractMeanAndStdForRandAndNorm(self, performanceIdx=1, nrOfReplicates=5):
        mean = np.zeros(4*len(self.selectedFolders))
        std = np.zeros(4*len(self.selectedFolders))
        startPerformanceValIdx = self.resultsTable[0].shape[1] // 2
        for i in range(len(self.selectedFolders)):
            table = self.randTable[i]
            mean[i*4] = table.iloc[nrOfReplicates, performanceIdx]
            mean[i*4+1] = table.iloc[nrOfReplicates, performanceIdx+startPerformanceValIdx]
            std[i*4] = table.iloc[nrOfReplicates+1, performanceIdx]
            std[i*4+1] = table.iloc[nrOfReplicates+1, performanceIdx+startPerformanceValIdx]
            table = self.resultsTable[i]
            mean[i*4+2] = table.iloc[nrOfReplicates, performanceIdx]
            mean[i*4+3] = table.iloc[nrOfReplicates, performanceIdx+startPerformanceValIdx]
            std[i*4+2] = table.iloc[nrOfReplicates+1, performanceIdx]
            std[i*4+3] = table.iloc[nrOfReplicates+1, performanceIdx+startPerformanceValIdx]
        return mean, std

    def extractMeanAndStd(self, performanceIdx=1, nrOfReplicates=5, plotOnlyRandom=False):
        mean = np.zeros(2*len(self.selectedFolders))
        std = np.zeros(2*len(self.selectedFolders))
        startPerformanceValIdx = self.resultsTable[0].shape[1] // 2
        for i in range(len(self.selectedFolders)):
            if plotOnlyRandom:
                table = self.randTable[i]
            else:
                table = self.resultsTable[i]
            mean[i*2] = table.iloc[nrOfReplicates, performanceIdx]
            mean[i*2+1] = table.iloc[nrOfReplicates, performanceIdx+startPerformanceValIdx]
            std[i*2] = table.iloc[nrOfReplicates+1, performanceIdx]
            std[i*2+1] = table.iloc[nrOfReplicates+1, performanceIdx+startPerformanceValIdx]
        return mean, std

    def addTestData(self, mean, std, idx, performanceIdx, tableList,
                    testMeanIdxName="test mean", testStdIdxName="test std"):
        mean, std = list(mean), list(std)
        for i, table in enumerate(tableList):
            indices = np.asarray(table.index)
            isTestMean = indices == testMeanIdxName
            isTestStd = indices == testStdIdxName
            meanTestP = table.iloc[isTestMean, performanceIdx].to_numpy()
            stdTestP = table.iloc[isTestStd, performanceIdx].to_numpy()
            mean.insert((idx+1)*i+idx, meanTestP)
            std.insert((idx+1)*i+idx, stdTestP)
        mean, std = np.asarray(mean), np.asarray(std)
        return mean, std

    def setYLabel(self, performanceIdx):
        performance = list(self.resultsTable[0].columns)[performanceIdx][6:]
        if performance == "Acc":
            yLabel = "Accuracy [%]"
        elif performance == "F1":
            yLabel = "F1-Score [%]"
        elif performance == "Auc":
            yLabel = "AUC"
        else:
            print("Warning the performance {} is not yet implemented.".format(performance))
            yLabel = ""
        return yLabel

    def calcTrainAndValDifferences(self, performanceIdx, compareRandAndNorm, nrOfReplicates=5,
                        correctPValues=True, printPValues=False, tukey=False):
        trainValues = []
        valValues = []
        startPerformanceValIdx = self.resultsTable[0].shape[1] // 2
        for table in self.resultsTable:
            trainValues.append(table.iloc[:nrOfReplicates, performanceIdx].to_numpy())
            valValues.append(table.iloc[:nrOfReplicates, performanceIdx+startPerformanceValIdx].to_numpy())
        trainPValues, valPValues, trainTStat, valTStat = [], [], [], []
        nrOfConditions = len(self.resultsTable)
        testCases = []
        group1 = []
        group2 = []
        if tukey:
            valTukey = self.doTukey(valValues, "validation")
            trainTukey = self.doTukey(trainValues, "train")
            tukey = []
            trainTukeyTxtLength = len(trainTukey[-1])
            for i in range(len(trainTukey)):
                currentLength = len(trainTukey[i])
                if currentLength < trainTukeyTxtLength:
                    difInLength = trainTukeyTxtLength - currentLength
                    additionalSpaces = " "*difInLength
                else:
                    additionalSpaces = ""
                tukey.append(trainTukey[i] + additionalSpaces + "    " + valTukey[i])
            tukey = "\n".join(tukey)
            if printPValues:
                print(tukey)
        for i, j in itertools.combinations(range(nrOfConditions), 2):
            TStat, pValues = st.ttest_rel(trainValues[i], trainValues[j])
            if np.isnan(pValues):
                pValues = 1
            trainPValues.append(pValues)
            trainTStat.append(TStat)
            TStat, pValues = st.ttest_rel(valValues[i], valValues[j])
            if np.isnan(pValues):
                pValues = 1
            valPValues.append(pValues)
            valTStat.append(TStat)
            testCases.append(self.selectedFolders[i]+" vs "+ self.selectedFolders[j])
            group1.append(self.selectedFolders[i])
            group2.append(self.selectedFolders[j])
        if correctPValues:
            trainPValues = list(multipletests(trainPValues, method='fdr_bh')[1])
            valPValues = list(multipletests(valPValues, method='fdr_bh')[1])
        pValueTable = {"group1":group1, "group2":group2,
                        "training p-values":trainPValues, "training T-stat:":trainTStat,
                       "validation p-values":valPValues, "validation T-stat:":valTStat}
        pValueTable = pd.DataFrame(pValueTable, index=testCases)
        if printPValues:
            print(pValueTable.to_string())
        return pValueTable

    def doTukey(self, values, name, alpha=0.05):
        if len(values) == 2:
            fvalue, pvalue = st.f_oneway(values[0], values[1])
        elif len(values) == 3:
            fvalue, pvalue = st.f_oneway(values[0], values[1], values[2])
        elif len(values) == 4:
            fvalue, pvalue = st.f_oneway(values[0], values[1], values[2], values[3])
        elif len(values) == 5:
            fvalue, pvalue = st.f_oneway(values[0], values[1], values[2], values[3], values[4])
        elif len(values) == 6:
            fvalue, pvalue = st.f_oneway(values[0], values[1], values[2], values[3], values[4], values[5])
        else:
            print("more than 6 groups is not yet implemented, {} != 2, 3, 4, 5, 6".format(len(values)))
            sys.exit()
        groupName = [[g]*len(v) for g, v in zip(self.selectedFolders, values)]
        values = np.concatenate(values)
        groupName = np.concatenate(groupName)
        m_comp = pairwise_tukeyhsd(endog=values, groups=groupName, alpha=0.05)
        textResult = name+"\n"+f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}\n"+str(m_comp)
        return textResult.split("\n")

def mainDivPredRandomization(performance="Acc", plotOnlyRandom=False, doMainFig=True,
                             baseResultsFolder = "Results/divEventData/manualCentres/",
                             addOtherTestWithBaseFolder=None, balanceData=False):
    performanceToIdxDict = {"F1":0, "Acc":1, "AUC":3}
    performanceIdx = performanceToIdxDict[performance]
    if performance != "AUC":
        minY = 50
    else:
        minY = 0.5
    if doMainFig:
        divEventPred = ["allTopos", "area", "topoAndBio", "lowCor0.7", "lowCor0.3", "topology"]
        addition = " main fig"
    else:
        divEventPred = ["lowCor0.3", "lowCor0.5", "lowCor0.7", "topology", "area"]
        addition = " sup low area"
    if not balanceData is None:
        if balanceData:
            balanceTxt = "bal1"
        else:
            balanceTxt = "bal0"
    else:
        balanceTxt = ""
    furtherFolder = "svm_k2h_combinedTable_l3f0n1c0{}ex0/".format(balanceTxt)
    setNames = ", ".join(divEventPred)
    if plotOnlyRandom:
        filenameToSave = baseResultsFolder + "div pred random results{} {} {}.png".format(addition, performance, setNames)
    else:
        filenameToSave = baseResultsFolder + "div pred {} results{} {} {}.png".format(balanceTxt, addition, performance, setNames)
    if addOtherTestWithBaseFolder:
        filenameToSave = filenameToSave.replace(".png", "_WithKtnTest.png")
    myBarPlotPlotter = BarPlotPlotter(baseResultsFolder, divEventPred,
                                      compareRandAndNorm=False,
                                      addOtherTestWithBaseFolder=addOtherTestWithBaseFolder,
                                      plotOnlyRandom=plotOnlyRandom,
                                      performanceIdx=performanceIdx,
                                      minY=minY,
                                      furtherFolder=furtherFolder,
                                      filenameToSave=filenameToSave)

def mainTopoPredRandomization(performance="Acc", doSpecial=False,
                              plotOnlyRandom=False, doMainFig=True,
                              balanceData=False,
                              excludeDivNeighbours=True, addOtherTestWithBaseFolder=None,
                              baseResultsFolder="Results/topoPredData/diff/manualCentres/",
                              selectedDivEventPred=None):
    performanceToIdxDict = {"F1":0, "Acc":4, "AUC":9}
    performanceIdx = performanceToIdxDict[performance]
    if performance != "AUC":
        minY = 30
    else:
        minY = 0.5
    if selectedDivEventPred is None:
        if doMainFig:
            divEventPred = ["allTopos", "bio", "topoAndBio", "lowCor0.7", "lowCor0.3", "topology"]# ["allTopos", "bio", "topoAndBio"]
        else:
            divEventPred = ["lowCor0.3", "lowCor0.5", "lowCor0.7", "topology", "bio"]
    else:
        divEventPred = selectedDivEventPred
    if excludeDivNeighbours:
        excludingTxt = "ex1"
    else:
        excludingTxt = "ex0"
    if not balanceData is None:
        if balanceData:
            balanceTxt = "bal1"
        else:
            balanceTxt = "bal0"
    else:
        balanceTxt = ""
    furtherFolder = "svm_k2h_combinedTable_l3f0n1c0{}{}/".format(balanceTxt, excludingTxt)
    if len(divEventPred) == 3:
        addition = " main fig"
    else:
        addition = " sup low area"
    setNames = ", ".join(divEventPred)
    if plotOnlyRandom:
        randFilename = "combinedResultsWithTestingOf_1000_randomizedRuns_{}.csv".format(excludingTxt)
        filenameToSave = baseResultsFolder + "topo random pred results {}{} {} {}.png".format(excludingTxt, addition, performance, setNames)
    else:
        filenameToSave = baseResultsFolder + "topo pred {} results {}{} {} {}.png".format(balanceTxt, excludingTxt, addition, performance, setNames)
    if addOtherTestWithBaseFolder:
        filenameToSave = filenameToSave.replace(".png", "_WithKtnTest.png")
    if doSpecial:
        minY = 0.5
        if doSpecial is True:
            # these values are printed by plotGivenFeatureSetRocCurves in MyScorer.py
            mean = np.asarray([0.7645570413063933, 0.777804868919271, 0.8211713750032892, 0.66540939310284, 0.7373431681712566, 0.6990939847659582, 0.6461213527485279, 0.8347198735203177, 0.839799406959718, 0.8408264614114299, 0.8397563798526405, 0.8296715078647696])
            std = np.asarray([0.02354825110390042, 0.03111001218656039, 0.01485431204178305, 0.02387554741313661, 0.025972683570405334, 0.03621836258368367, 0.04859439598667418, 0.02678331651271472, 0.020176032220297544, 0.026778787924009884, 0.022729253518084462, 0.0242913495769443])
        else:
            assert len(doSpecial) == 2, "The given special "
            mean, std = doSpecial
        #                           class 0    class 1    class 2
        coloryByHtml = ["#5288EA", "#2EF8F9", "#F59C24", "#E550AB"] # ["#00ff00", "#91e3e3", "#F59C24", "#da76b3"] # ["#5288EA", "#2EF8F9", "#E550AB"] last class 0, 1, 2
        idx = 4
        doSpecial = {"mean":mean, "std":std, "coloryByHtml":coloryByHtml, "idx":idx}
        filenameToSave = baseResultsFolder + "topo pred results detailed auc allTopos, bio, topoAndBio.png"
    myBarPlotPlotter = BarPlotPlotter(baseResultsFolder, divEventPred,
                                      compareRandAndNorm=False,
                                      resultsTestFilename=None,
                                      addOtherTestWithBaseFolder=addOtherTestWithBaseFolder,
                                      plotOnlyRandom=plotOnlyRandom,
                                      furtherFolder=furtherFolder,
                                      performanceIdx=performanceIdx,
                                      minY=minY, doSpecial=doSpecial,
                                      filenameToSave=filenameToSave)

def main():
    doDivPredPlots = True
    plotRandomResults = False
    addOtherTestWithBaseFolder = False # set True to include _WithKtnTest
    if plotRandomResults:
        if doDivPredPlots:
            mainDivPredRandomization(performance="Acc", plotOnlyRandom=plotRandomResults)
        else:
            mainTopoPredRandomization(performance="Acc", plotOnlyRandom=plotRandomResults)
            # mainTopoPredRandomization(performance="Acc", plotOnlyRandom=plotRandomResults,
            #                           excludeDivNeighbours=False)
    else:
        if doDivPredPlots:
            if addOtherTestWithBaseFolder:
                addOtherTestWithBaseFolder = "Results/ktnDivEventData/manualCentres/"
            else:
                addOtherTestWithBaseFolder = None
            mainDivPredRandomization(performance="Acc", doMainFig=True, baseResultsFolder="Results/divEventData/manualCentres/",
                                     addOtherTestWithBaseFolder=addOtherTestWithBaseFolder)
            # for p, isMain in zip(["Acc", "AUC", "Acc", "AUC"], [True, True, False, False]):
            #     mainDivPredRandomization(performance=p, doMainFig=isMain, baseResultsFolder="Results/divEventData/manualCentres/",
            #                              addOtherTestWithBaseFolder=addOtherTestWithBaseFolder)
        else:
            if addOtherTestWithBaseFolder:
                addOtherTestWithBaseFolder = "Results/ktnTopoPredData/diff/manualCentres/"
            else:
                addOtherTestWithBaseFolder = None
            # for p, isMain in zip(["Acc", "AUC", "Acc", "AUC"], [True, True, False, False]):
            #     mainTopoPredRandomization(performance=p, doMainFig=isMain,
            #                               addOtherTestWithBaseFolder=addOtherTestWithBaseFolder)
            # mainTopoPredRandomization(performance="AUC", doMainFig=False, doSpecial=True)
            mainTopoPredRandomization(performance="Acc", doMainFig=True, balanceData=False,
                                      excludeDivNeighbours=True,
                                      addOtherTestWithBaseFolder=addOtherTestWithBaseFolder)

if __name__ == '__main__':
    main()
