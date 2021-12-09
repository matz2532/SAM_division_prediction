import numpy as np
import pandas as pd
import scipy.stats as st
import sys

class GeneralDataAnalyser (object):

    # save the number of labels per tissue
    def __init__(self, featureTable=None, labelTable=None):
        self.featureTable = featureTable
        self.labelTable  = labelTable

    def AnalyseLabelDistributionOverTissues(self, labelTable=None, plantNameColIdx=0,
                                            timePointColIdx=1, labelColIdx=-1,
                                            excludeNameTimePointPairList=None,
                                            saveResultsToFilename=None,
                                            printResults=False):
        if labelTable is None:
            labelTable = self.labelTable
        assert not labelTable is None, "The label table needs to be given as a argument to AnalyseLabelDistributionOverTissues or in the class initalisation. labelTable is None"
        tissuesNamesLabelsAndCounts = self.extractTissuesNamesLabelsAndCounts(labelTable, plantNameColIdx, timePointColIdx, labelColIdx)
        tissueNames, allTissuesUniqueLabels, allTissuesUniqueCounts = tissuesNamesLabelsAndCounts
        countArray = self.reorderCountsToArray(allTissuesUniqueLabels, allTissuesUniqueCounts)
        if countArray.shape[1] == 2:
            percentageOfLabelOnePerTissue = [100*countsOfTissue[1]/np.sum(countsOfTissue) for countsOfTissue in countArray]
            meanPercentageOfLabelOnes = np.mean(percentageOfLabelOnePerTissue)
            stdPercentageOfLabelOnes = np.std(percentageOfLabelOnePerTissue)
            percentageOfLabelOnePerTissue.append("{}+-{}".format(meanPercentageOfLabelOnes, stdPercentageOfLabelOnes))
            countsLabelZero = list(countArray[:, 0])
            countsLabelZero.append(np.NaN)
            countsLabelOnes = list(countArray[:, 1])
            countsLabelOnes.append(np.NaN)
            tissueNameStringList = ["{}_{}".format(name, timePoint) for name, timePoint in tissueNames]
            tissueNameStringList.append("mean and std of percentage label 1")
            plantNames = [i[0] for i in tissueNames]
            plantNames.append(np.NaN)
            timePoints = [i[1] for i in tissueNames]
            timePoints.append(np.NaN)
            countTable = pd.DataFrame({"tissue name":tissueNameStringList,
                                       "counts of label 0":countsLabelZero,
                                       "counts of label 1":countsLabelOnes,
                                       "percentage of label 1":percentageOfLabelOnePerTissue,
                                       "plant name":plantNames,
                                       "time point":timePoints})
            if saveResultsToFilename:
                countTable.to_csv(saveResultsToFilename, index=False)
            if printResults:
                print(countTable.to_string())
        elif countArray.shape[1] == 3:
            tissueNameAsRowName = ["{}".format(name) for name, timePoint in tissueNames]
            colNames = ["label {}".format(i) for i in range(countArray.shape[1])]
            countArrayDf = pd.DataFrame(countArray, columns=colNames, index=tissueNameAsRowName)
            print(countArrayDf.to_string())
            print("sum of", colNames)
            print("      ",np.sum(countArray, axis=0))
        else:
            print("Not yet implemented analysis with more than two label types.")

    def extractTissuesNamesLabelsAndCounts(self, labelTable, plantNameColIdx,
                                           timePointColIdx, labelColIdx):
        plantNames = labelTable.iloc[:, plantNameColIdx].to_numpy()
        timePoints = labelTable.iloc[:, timePointColIdx].to_numpy()
        labels = labelTable.iloc[:, labelColIdx].to_numpy()
        uniquePlantNames = np.unique(plantNames)
        tissueNames = []
        allTissuesUniqueLabels, allTissuesUniqueCounts= [], []
        for currentPlantName in uniquePlantNames:
            isPlantName = np.isin(plantNames, currentPlantName)
            uniqueTimePoints = np.unique(timePoints[isPlantName])
            for currentTimePoint in uniqueTimePoints:
                isTimePoint = np.isin(timePoints, currentTimePoint)
                isTissue = isPlantName & isTimePoint
                labelsOfTissue = labels[isTissue]
                uniqueLabel, count = np.unique(labelsOfTissue, return_counts=True)
                tissueNames.append([currentPlantName, currentTimePoint])
                allTissuesUniqueLabels.append(uniqueLabel)
                allTissuesUniqueCounts.append(count)
        return tissueNames, allTissuesUniqueLabels, allTissuesUniqueCounts

    def reorderCountsToArray(self, allTissuesUniqueLabels, allTissuesUniqueCounts):
        uniqueLabels = np.unique(allTissuesUniqueLabels)
        shape = (len(allTissuesUniqueLabels), len(uniqueLabels))
        countArray = np.zeros(shape)
        for i in range(shape[0]):
            currentUniqueLabels = allTissuesUniqueLabels[i]
            if np.all(uniqueLabels == currentUniqueLabels):
                countArray[i, :] = allTissuesUniqueCounts[i]
            else:
                currenCounts = allTissuesUniqueCounts[i]
                for j, label in enumerate(currentUniqueLabels):
                    labelIdx = np.where(uniqueLabels == label)[0]
                    countArray[i, labelIdx] = currenCounts[j]
                print("uniqueLabels are not equal to currentUniqueLabels", uniqueLabels == currentUniqueLabels, uniqueLabels, currentUniqueLabels)
                print("This branch is not yet tested.")
                print("Please verify your results.")
        return countArray

    def OrderPercentagesOfLabelOne(self, table, colIdxToDisplay=3, plantNameColIdx=4,
                                   timePointColIdx=5, saveResultsToFilename=None,
                                   printResults=False):
        columnToDisplay = table.iloc[:-1, colIdxToDisplay].to_numpy()
        plantNames = table.iloc[:-1, plantNameColIdx].to_numpy()
        timePoints = table.iloc[:-1, timePointColIdx].to_numpy()
        uniquePlantNames = np.unique(plantNames)
        allUniqueTimePoints = np.unique(timePoints)
        orderedValues = np.full((len(uniquePlantNames), len(allUniqueTimePoints)), np.NaN)
        for i, currentPlantName in enumerate(uniquePlantNames):
            isPlantName = np.isin(plantNames, currentPlantName)
            uniqueTimePoints = np.unique(timePoints[isPlantName])
            for currentTimePoint in uniqueTimePoints:
                isTimePoint = np.isin(timePoints, currentTimePoint)
                isTissue = isPlantName & isTimePoint
                value = float(columnToDisplay[isTissue][0])
                rowIdx = np.where(allUniqueTimePoints == currentTimePoint)[0]
                orderedValues[i, rowIdx] = value
        columns = ["T{}".format(int(t)) for t in allUniqueTimePoints]
        orderedValuesDf = pd.DataFrame(orderedValues, columns=columns, index=uniquePlantNames)
        if saveResultsToFilename:
            orderedValuesDf.to_csv(saveResultsToFilename)
        if printResults:
            print(orderedValuesDf.to_string())

    def CheckSignificanceBetweenTables(self, firstTable, secondTable, selectedColumn,
                                       excludeLastEntry=True, testName="unpaired t-test",
                                       tissueNamesToExclude=None):
        firstValues = firstTable.iloc[:, selectedColumn].to_numpy()
        secondValues = secondTable.iloc[:, selectedColumn].to_numpy()
        if excludeLastEntry:
            firstValues = firstValues[:-1]
            secondValues = secondValues[:-1]
        if tissueNamesToExclude:
            firstTissueNames = firstTable.iloc[:, 0]
            secondTissueNames = secondTable.iloc[:, 0]
            if excludeLastEntry:
                firstTissueNames = firstTissueNames[:-1]
                secondTissueNames = secondTissueNames[:-1]
            firstValues = firstValues[np.isin(firstTissueNames, tissueNamesToExclude, invert=True)]
            secondValues = secondValues[np.isin(secondTissueNames, tissueNamesToExclude, invert=True)]
        firstValues = firstValues.astype(float)
        secondValues = secondValues.astype(float)
        if testName == "unpaired t-test":
            testResults = st.ttest_ind(firstValues, secondValues)
        else:
            print("The test {} is not yet implemented. Choose one of the following as the testName: unpaired t-test".format(testName))
            return None
        print(testResults)

    def combineResultsPerformances(self, resultsTableFolder="Results/divEventData/manualCentres/adjusted div Pred/{}/svm_k2h_combinedTable_l3f0n1c0ex0/", baseResultsFilename="resultsWithTesting.csv",
                                   filenameToSave="Results/divEventData/manualCentres/adjusted div Pred/combinedResultsOfFeatureSets.csv",
                                   featureSets=["allTopos", "area", "topoAndBio", "lowCor0.7", "lowCor0.5", "topology"],
                                   additionalTableFolder="Results/ktnDivEventData/manualCentres/adjusted div Pred/{}/svm_k2h_combinedTable_l3f0n1c0ex0/",
                                   addAdditionalTableDataAtTheEnd=True, addedRowName="ktn testing"):
        allTables = []
        for set in featureSets:
            folder = resultsTableFolder.format(set)
            table = pd.read_csv(folder + baseResultsFilename, header=None)
            allTables.append(table)
        if not additionalTableFolder is None:
            for i, set in enumerate(featureSets):
                folder = additionalTableFolder.format(set)
                tableToAppend = pd.read_csv(folder + baseResultsFilename, header=None).iloc[1, 1:]
                assert len(tableToAppend.shape) == 1, "The added table needs to be one dimensional. Other versions are not yet implemented. {} != 1".format(len(tableToAppend.shape))
                combinedTable = allTables[i]
                if addAdditionalTableDataAtTheEnd:
                    nrOfCols = combinedTable.shape[1]
                    nrOfEmptyCols = nrOfCols - len(tableToAppend)
                    if nrOfEmptyCols > 0:
                        fillerSeries = pd.Series(np.full(nrOfEmptyCols, np.NaN))
                        fillerSeries.iloc[0] = addedRowName
                        tableToAppend = pd.concat([fillerSeries, tableToAppend], ignore_index=True)
                allTables[i] = combinedTable.append(tableToAppend, ignore_index=True)
        concatenatedTable = pd.concat(allTables)
        print("saving", filenameToSave)
        concatenatedTable.to_csv(filenameToSave, index=False, header=False)

def main():
    combineDivPredResults = True
    combineTopoPredResults = True
    combineResultsBaseName = "combinedResultsOfFeatureSets.csv"
    divBaseFolder = "Results/divEventData/manualCentres/adjusted div Pred/"
    divArgs = {"resultsTableFolder" : divBaseFolder + "{}/svm_k2h_combinedTable_l3f0n1c0ex0/",
            "filenameToSave" : divBaseFolder + combineResultsBaseName,
            "featureSets" : ["allTopos", "area", "topoAndBio", "lowCor0.7", "lowCor0.5", "topology"],
            "additionalTableFolder" : "Results/ktnDivEventData/manualCentres/adjusted div Pred/{}/svm_k2h_combinedTable_l3f0n1c0ex0/"}
    topoBaseFolder = "Results/topoPredData/diff/manualCentres/"
    topoArgs = {"resultsTableFolder" : topoBaseFolder + "{}/svm_k2h_combinedTable_l3f0n1c0ex1/",
            "filenameToSave" : topoBaseFolder + combineResultsBaseName,
            "featureSets" : ["allTopos", "bio", "topoAndBio", "lowCor0.7", "lowCor0.5", "topology"],
            "additionalTableFolder" : "Results/ktnTopoPredData/diff/manualCentres/{}/svm_k2h_combinedTable_l3f0n1c0ex1/"}
    myGeneralDataAnalyser = GeneralDataAnalyser()
    if combineDivPredResults:
        myGeneralDataAnalyser.combineResultsPerformances(**divArgs)
    if combineTopoPredResults:
        myGeneralDataAnalyser.combineResultsPerformances(**topoArgs)

def mainAnalyseNumbersOfDividingCells(baseFolderName="Data/{}/divEventData/manualCentres/",
                analyseFolderWithExtension=["WT", "ktn"],
                tissueNamesToExclude=["P6_1", "P6_2", "P8_2", "P8_3", "P10_2"],
                redoTables=True):
    myGeneralDataAnalyser = GeneralDataAnalyser()
    if redoTables:
        for folderExtension in analyseFolderWithExtension:
            folder = baseFolderName.format(folderExtension)
            labelTableFilename = folder + "topology/combinedLabels.csv"
            labelTable = pd.read_csv(labelTableFilename)
            saveResultsToFilename = folder + "labelCountPerTissue.csv"
            myGeneralDataAnalyser.AnalyseLabelDistributionOverTissues(labelTable=labelTable,
                                                saveResultsToFilename=saveResultsToFilename)
            table = pd.read_csv(saveResultsToFilename)
            saveResultsToFilename = folder + "dividingCellsPercentage.csv"
            myGeneralDataAnalyser.OrderPercentagesOfLabelOne(table, saveResultsToFilename=saveResultsToFilename)
    tableNames = [baseFolderName.format(folderExtension) + "labelCountPerTissue.csv" for folderExtension in analyseFolderWithExtension]
    tables = [pd.read_csv(tableName) for tableName in tableNames]
    myGeneralDataAnalyser.CheckSignificanceBetweenTables(tables[0], tables[1], selectedColumn=3,
                                                        tissueNamesToExclude=tissueNamesToExclude)

def mainAnalyseNumbersOfLocalTopologyPrediction():
    analyseFolderWithExtension = ["ktn/topoPredData", "WT/topoPredData"]
    baseFolderName = "Data/{}/diff/manualCentres/"
    myGeneralDataAnalyser = GeneralDataAnalyser()
    redoTables = True
    if redoTables:
        for folderExtension in analyseFolderWithExtension:
            print(folderExtension)
            folder = baseFolderName.format(folderExtension)
            labelTableFilename = folder + "topology/combinedLabels.csv"
            labelTable = pd.read_csv(labelTableFilename)
            saveResultsToFilename = folder + "labelCountPerTissue.csv"
            myGeneralDataAnalyser.AnalyseLabelDistributionOverTissues(labelTable=labelTable,
                                                saveResultsToFilename=saveResultsToFilename)

if __name__ == '__main__':
    # main()
    mainAnalyseNumbersOfDividingCells()
    # mainAnalyseNumbersOfLocalTopologyPrediction()
