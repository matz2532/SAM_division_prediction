import numpy as np
import pandas as pd
import pickle
import sys

def mainPrintXOutOfPossibleNonDivCellAfterPropagation():
    # check acutal out of potential non-dividing cells which were compared in propagation
    baseDataFolder = "Data/WT/divEventData/manualCentres/topology/"
    baseResultsFolder="Results/DivAndTopoApplication/"
    plantNames = ["P2", "P9"]
    fullLabelTable = pd.read_csv(baseDataFolder+"/combinedLabels.csv")
    fullPredictedFeatureOfNonDivCellsTable = []
    for plant in plantNames:
        predictedFeatureTable = np.load(baseResultsFolder+plant+"/predFeatures.npy")
        fullPredictedFeatureOfNonDivCellsTable.append(predictedFeatureTable)
    isPlant = np.isin(fullLabelTable["plant"], plantNames)
    fullLabelTable = fullLabelTable.iloc[isPlant, :]
    possibleNonDiv = np.sum(fullLabelTable["label"] == 0)
    fullPredictedFeatureOfNonDivCellsTable = np.concatenate(fullPredictedFeatureOfNonDivCellsTable, axis=0)
    actualNonDiv = len(fullPredictedFeatureOfNonDivCellsTable)
    print(f"{actualNonDiv} of the possible {possibleNonDiv} non-dividing cells in the observed topology were also non-dividing in the predicted topology")

def loadAndExtractPerformance(baseResultsFolder, featureSet, kernel="rbf", excludeTxt="0",
                              modelTypeProp="svm_k{}h_combinedTable_l3f0n1c0bal0ex{}",
                              resultFilename="results.csv",
                              columnToSelect="val Acc",
                              indexToUse="mean"):
    assert kernel == "rbf" or kernel == "linear", "The kernel needs to be 'rbf' or 'linear'."
    kernelTxt = "2" if kernel == "rbf" else  "1"
    filename = f"{baseResultsFolder}{featureSet}/{modelTypeProp.format(kernelTxt, excludeTxt)}/{resultFilename}"
    table = pd.read_csv(filename, index_col=0)
    return table.loc[indexToUse, columnToSelect]

def calcMeanPerformanceOf(baseResultsFolder, givenSets, kernel="rbf", excludeTxt="0", resultFilename="results.csv"):
    performances = []
    for featureSet in givenSets:
        performances.append(loadAndExtractPerformance(baseResultsFolder, featureSet, kernel=kernel, excludeTxt=excludeTxt))
    print(performances)
    return np.mean(performances)

def printDecisionBetweenRbfAndLinearKernelFor(baseResultsFolder="Results/", checkDivEventPred=True, performanceThreshold=1.5):
    # performanceThreshold means in case linear kernel performs only 1.5% worse than rbf kernel our threshold is met to choose the linear kernel
    if checkDivEventPred:
        scenarioTxt = "division event pred"
        predTypeExtension = "divEventData/manualCentres/"
        givenSets = ["allTopos", "area", "topoAndBio"]#, "topology", "lowCor0.3", "lowCor0.7"]
        excludeTxt = "0"
    else:
        scenarioTxt = "local topology pred"
        predTypeExtension = "topoPredData/diff/manualCentres/"
        givenSets = ["allTopos", "bio", "topoAndBio"]#, "topology", "lowCor0.3", "lowCor0.7"]
        excludeTxt = "1"
    baseResultsFolder += predTypeExtension
    meanRbfKernelValidationAccuracy = calcMeanPerformanceOf(baseResultsFolder, givenSets, kernel="rbf", excludeTxt=excludeTxt,  resultFilename="resultsWithTesting.csv")
    meanLinearKernelValidationAccuracy = calcMeanPerformanceOf("Results/topoPredData/concatPlusDiff/manualCentres/", givenSets, kernel="linear", excludeTxt=excludeTxt)
    meanDifference = meanRbfKernelValidationAccuracy - meanLinearKernelValidationAccuracy
    if meanDifference < performanceThreshold:
        kernelToUseTxt = "linear"
    else:
        kernelToUseTxt = "rbf"
    print(f"Use the kernel {kernelToUseTxt} for {scenarioTxt} as {meanRbfKernelValidationAccuracy} - {meanLinearKernelValidationAccuracy} = {meanDifference} < {performanceThreshold} (perform rbf - linear < threshold).")

def mainDecideBetweenRbfAndLinearKernelForDivsionAndTopoPredictionModels():
    # printDecisionBetweenRbfAndLinearKernelFor(checkDivEventPred=True)
    printDecisionBetweenRbfAndLinearKernelFor(checkDivEventPred=False)

if __name__== "__main__":
    # mainPrintXOutOfPossibleNonDivCellAfterPropagation()
    mainDecideBetweenRbfAndLinearKernelForDivsionAndTopoPredictionModels()
