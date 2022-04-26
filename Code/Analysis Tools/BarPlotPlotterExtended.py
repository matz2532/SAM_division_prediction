import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from BarPlotPlotter import BarPlotPlotter
from pathlib import Path
from PValueToLetterConverter import PValueToLetterConverter
from statsmodels.stats.multitest import multipletests

class BarPlotPlotterExtended (BarPlotPlotter):

    resultsTable=None
    plotOnlyRandom=False

    def __init__(self, scenarioResultsFolders, selectedFeatureSetFolders,
                 addOtherTestWithBaseFolder=None,
                 furtherFolder="svm_k2h_combinedTable_l3f0n1c0bal1ex0/",
                 resultFilename="results.csv", resultsTestFilename="resultsWithOnlyTesting.csv", performanceIdx=1,
                 filenameToSave="", fontSize=18,
                 minY=0, doSpecial=False, nrOfReplicates=6):
        self.scenarioIdxToNameDict = {0 : "WT SAM", 1 : "ktn1-2 SAM", 2 : "WT floral meristem", 3 : "ktn floral meristem"}
        self.scenarioResultsFolders = scenarioResultsFolders
        self.selectedFeatureSetFolders = selectedFeatureSetFolders
        self.addOtherTestWithBaseFolder = addOtherTestWithBaseFolder
        self.furtherFolder = furtherFolder
        self.resultFilename = resultFilename
        self.resultsTestFilename = resultsTestFilename
        self.performanceIdx = performanceIdx
        self.fontSize = fontSize
        self.minY = minY
        self.doSpecial = doSpecial
        self.filenameToSave = filenameToSave
        self.nrOfReplicates = nrOfReplicates
        self.loadFiles()
        self.createFigures(self.performanceIdx, minY=self.minY)

    def loadFiles(self):
        self.testResultTables = []
        for baseResultsFolder in self.scenarioResultsFolders:
            table = self.loadTables(baseResultsFolder, addFurtherFolder=True,
                                    addSpecificNameSuffix=self.resultsTestFilename)
            self.testResultTables.append(table)

    def createFigures(self, performanceIdx=1, minY=0, baseScenarioIdx=0,
                      printPValues=False):
        x_pos, mean, std, colors = self.setupData(performanceIdx)
        statisticsLetters = ""
        pValueTable = None
        self.doStatistics(performanceIdx, baseScenarioIdx)
        self.plotFigure(x_pos, mean, std, colors, performanceIdx, minY, fontSize=self.fontSize)

    def setupData(self, performanceIdx):
        mean, std = self.extractMeanAndStd(performanceIdx)
        idx = 4
        coloryByHtml = ["#ff5800", "#d1008f", "#2E75B6", "#548235"]
        x_pos = np.arange(len(mean))
        x_pos += np.arange(len(mean)) // idx
        colors = np.full(len(x_pos), coloryByHtml[0])
        colors[[i%idx==1 for i in range(len(mean))]] = coloryByHtml[1]
        colors[[i%idx==2 for i in range(len(mean))]] = coloryByHtml[2]
        colors[[i%idx==3 for i in range(len(mean))]] = coloryByHtml[3]
        return x_pos, mean, std, colors

    def extractMeanAndStd(self, performanceIdx):
        mean = []
        std = []
        for i, featureSet in enumerate(self.selectedFeatureSetFolders):
            for j, baseResultsFolder in enumerate(self.scenarioResultsFolders):
                table = self.testResultTables[j][i]
                currentMean, currentStd = self.getTestTablesMeanAndStd(table, performanceIdx)
                mean.append(currentMean)
                std.append(currentStd)
        return mean, std

    def getTestTablesMeanAndStd(self, table, performanceIdx,
                                testMeanIdxName="test mean", testStdIdxName="test std"):
        indices = np.asarray(table.index)
        isTestMean = indices == testMeanIdxName
        isTestStd = indices == testStdIdxName
        meanTestP = table.iloc[isTestMean, performanceIdx].to_numpy()
        stdTestP = table.iloc[isTestStd, performanceIdx].to_numpy()
        return meanTestP[0], stdTestP[0]

    def doStatistics(self, performanceIdx, baseScenarioIdx):
        setsPerofmancesOfBaseScenario = self.testResultTables[baseScenarioIdx]
        differencesBetweenTestScenariosDf = []
        for featureSetIdx, featureSetName in enumerate(self.selectedFeatureSetFolders):
            scenarioResultTables = self.selectTestScenarioResultsOfFeature(featureSetIdx)
            statisticsOfComparison = self.testAllPerformancesVsIdx(scenarioResultTables, baseScenarioIdx, performanceIdx, featureSetName)
            differencesBetweenTestScenariosDf.append(statisticsOfComparison)
        betweenTestScenariosTableName = Path(self.filenameToSave).with_name(Path(self.filenameToSave).stem + "_betweenTestScenariosPValues.csv")
        differencesBetweenTestScenariosDf = pd.concat(differencesBetweenTestScenariosDf)
        differencesBetweenTestScenariosDf.to_csv(betweenTestScenariosTableName)

    def selectTestScenarioResultsOfFeature(self, featureSetIdx):
        scenarioResultTables = []
        for allFeaturesResultsOfScenario in self.testResultTables:
            scenarioResultTables.append(allFeaturesResultsOfScenario[featureSetIdx])
        return scenarioResultTables

    def testAllPerformancesVsIdx(self, scenarioResultTables, baseScenarioIdx, performanceIdx,
                                 featureSetName="", correctPValues=True, indexName="test tissue",
                                 pValueColumnName="p-values", tStatsColumnName="T-stat"):
        allPValues, allTStats, testCases, group1, group2, usedStatisticalMethod = [], [], [], [], [], []
        baseScenarioName = self.scenarioIdxToNameDict[baseScenarioIdx]
        baseScenarioPerformances = self.selectPerformancesFromTableList([scenarioResultTables[baseScenarioIdx]], performanceIdx, indexName)[0]
        for versusScenarioIdx in range(len(self.testResultTables)):
            if versusScenarioIdx != baseScenarioIdx:
                versusScenarioName = self.scenarioIdxToNameDict[versusScenarioIdx]
                versusScenarioPerformances = self.selectPerformancesFromTableList([scenarioResultTables[versusScenarioIdx]], performanceIdx, indexName)[0]
                pValue, stat, statsMethod = self.pairwiseComparisonTest(baseScenarioPerformances, versusScenarioPerformances)
                allPValues.append(pValue)
                allTStats.append(stat)
                testCases.append(f"{baseScenarioName} vs {versusScenarioName} {featureSetName}")
                group1.append(f"{baseScenarioName} {featureSetName}")
                group2.append(f"{versusScenarioName} {featureSetName}")
                usedStatisticalMethod.append(statsMethod)
        if correctPValues:
            allPValues = list(multipletests(allPValues, method='fdr_bh')[1])
        pValueTable = {"group1" : group1, "group2" : group2,
                       pValueColumnName : allPValues,
                       tStatsColumnName : allTStats,
                       "statistical method" : usedStatisticalMethod}
        pValueTable = pd.DataFrame(pValueTable, index=testCases)
        return pValueTable

def mainDivPredTestComparisons(performance="Acc", doMainFig=True,
                             scenarioResultsFolders = ["Results/divEventData/manualCentres/",
                                                  "Results/ktnDivEventData/manualCentres/",
                                                  "Results/floral meristems/WT/divEventData/manualCentres/",
                                                  "Results/floral meristems/ktn/divEventData/manualCentres/"],
                             balanceData=False, fontSize=18,
                             savePlotFolder=None, resultsTestFilename="resultsWithOnlyTesting.csv"):
    performanceToIdxDict = {"F1":0, "Acc":1, "AUC":3}
    performanceIdx = performanceToIdxDict[performance]
    if performance != "AUC":
        minY = 50
    else:
        minY = 0.5
    if doMainFig:
        featureSetNames = ["allTopos", "area", "topoAndBio", "lowCor0.3", "topology"]
        addition = " main fig"
    else:
        featureSetNames = ["lowCor0.3", "lowCor0.5", "lowCor0.7", "topology", "area"]
        addition = " sup low area"
    if not balanceData is None:
        if balanceData:
            balanceTxt = "bal1"
        else:
            balanceTxt = "bal0"
    else:
        balanceTxt = ""
    furtherFolder = "svm_k2h_combinedTable_l3f0n1c0{}ex0/".format(balanceTxt)
    setNames = ", ".join(featureSetNames)
    if savePlotFolder is None:
        savePlotFolder = scenarioResultsFolders[0]
    filenameToSave = savePlotFolder + "test div pred {} results{} {} {}.png".format(balanceTxt, addition, performance, setNames)
    myBarPlotPlotter = BarPlotPlotterExtended(scenarioResultsFolders, featureSetNames,
                                      performanceIdx=performanceIdx,
                                      minY=minY, fontSize=fontSize,
                                      furtherFolder=furtherFolder,
                                      filenameToSave=filenameToSave,
                                      resultsTestFilename=resultsTestFilename)

def mainTopoPredTestComparisons(performance="Acc", doMainFig=True,
                              scenarioResultsFolders = ["Results/topoPredData/diff/manualCentres/",
                                                   "Results/ktnTopoPredData/diff/manualCentres/",
                                                   "Results/floral meristems/WT/topoPredData/diff/manualCentres/",
                                                   "Results/floral meristems/ktn/topoPredData/diff/manualCentres/"],
                              balanceData=False, fontSize=18,
                              excludeDivNeighbours=True,
                              selectedFeatureSetNames=None,
                              savePlotFolder=None, resultsTestFilename="resultsWithOnlyTesting.csv"):
    performanceToIdxDict = {"F1":0, "Acc":4, "AUC":9}
    performanceIdx = performanceToIdxDict[performance]
    if performance != "AUC":
        minY = 30
    else:
        minY = 0.5
    if selectedFeatureSetNames is None:
        if doMainFig:
            featureSetNames = ["allTopos", "bio", "topoAndBio", "lowCor0.3", "topology"]# ["allTopos", "bio", "topoAndBio"]
        else:
            featureSetNames = ["lowCor0.3", "lowCor0.5", "lowCor0.7", "topology", "bio"]
    else:
        featureSetNames = selectedFeatureSetNames
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
    if len(featureSetNames) == 3:
        addition = " main fig"
    else:
        addition = " sup low area"
    setNames = ", ".join(featureSetNames)
    if savePlotFolder is None:
        savePlotFolder = scenarioResultsFolders[0]
    filenameToSave = savePlotFolder + "test topo pred {} results {}{} {} {}.png".format(balanceTxt, excludingTxt, addition, performance, setNames)
    myBarPlotPlotter = BarPlotPlotterExtended(scenarioResultsFolders, featureSetNames,
                                      resultsTestFilename=resultsTestFilename,
                                      furtherFolder=furtherFolder,
                                      performanceIdx=performanceIdx,
                                      minY=minY, fontSize=fontSize,
                                      filenameToSave=filenameToSave)

def main():
    mainDivPredTestComparisons(savePlotFolder="Results/MainFigures/Fig 2 alternative/", fontSize=24)
    mainTopoPredTestComparisons(savePlotFolder="Results/MainFigures/Fig 3 alternative/", fontSize=24)

if __name__ == '__main__':
    main()
