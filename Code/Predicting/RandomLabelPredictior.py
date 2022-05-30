import numpy as np
import pandas as pd
import pickle
import sys
import warnings
warnings.simplefilter("ignore", UserWarning)

from pathlib import Path
from PredictonManager import PredictonManager

class RandomLabelPredictior (object):

    # predict models based on label randomization (evaluate train and validation performance and repeat)
    def __init__(self, featureSet, nrOfRandomRuns=3, seed=42, recreateRadomLabels=True,
                 runModel=True, excludeDividingNeighbours=True, divEventPred=False,
                 printResultsFolder=False, printEndResult=False, includeTesting=False):
        self.excludeDividingNeighbours = excludeDividingNeighbours # "combinedLabelsChecked{}.csv"
        self.printResultsFolder = printResultsFolder
        self.printEndResult = printEndResult
        self.divEventPred = divEventPred
        self.includeTesting = includeTesting
        assert self.includeTesting is True or self.includeTesting is False, "includeTesting needs to be True or False, {} is not True or False".format(includeTesting)
        self.setBaseParameters(featureSet)
        np.random.seed(seed)
        if recreateRadomLabels:
            self.createRandomLabels(nrOfRandomRuns)
        if runModel:
            self.runModels(nrOfRandomRuns, self.basicResultsFolder)
        self.combineRandomResults(nrOfRandomRuns, self.basicResultsFolder)

    def setBaseParameters(self, featureSet):
        if self.divEventPred is True:
            insert = "divEventData/"
            self.useOnlyTwo = True
        else:
            insert = "topoPredData/diff/"
            self.useOnlyTwo = False
        assert featureSet in ["area", "bio", "topology", "topoAndBio", "allTopos", "lowCor0.3", "lowCor0.5", "lowCor0.7"], "featureSet needs to be area, bio, topology, allTopos, or topoAndBio, featureSet is {}".format(featureSet)
        self.setFeatureAndLabelFolder = "Data/WT/{}manualCentres/{}/".format(insert, featureSet)
        self.basicResultsFolder = "Results/{}manualCentres/{}/".format(insert, featureSet)
        self.givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(featureSet)
        self.basicLabelName = "randomisedCombinedLabels{}.csv"
        self.labelName = "combinedLabels.csv"
        if self.excludeDividingNeighbours:
            excludedValue = 1
            modelFolderName = "svm_k1h_combinedTable_l3f0n1c0bal0ex1/"
        else:
            excludedValue = 0
            modelFolderName = "svm_k1h_combinedTable_l3f0n1c0bal0ex0/"
        externalModelFilename = f"{self.basicResultsFolder}{modelFolderName}testModel.pkl"
        with open(externalModelFilename, "rb") as fh:
            externalModel = pickle.load(fh)
        self.hyperParameters = externalModel.GetHyperParameters()

    def createRandomLabels(self, nrOfRandomRuns):
        print("Shuffeling labels")
        labels = pd.read_csv(self.setFeatureAndLabelFolder + self.labelName)
        idx = np.arange(len(labels))
        for i in range(nrOfRandomRuns):
            np.random.shuffle(idx)
            newLabels = labels.copy()
            newLabels.iloc[:, -1] = labels.iloc[idx, -1].to_numpy()
            newFilename = self.setFeatureAndLabelFolder + self.basicLabelName.format(i+1)
            newLabels.to_csv(newFilename, index=False)

    def runModels(self, nrOfRandomRuns, basicResultsFolder):
        for i in range(1, nrOfRandomRuns+1):
            if i % 20 == 0:
                print("currentRandomSample: {}/{}".format(i, nrOfRandomRuns))
            labelNameCurrentRand = self.basicLabelName.format(i)
            resultsFolder = basicResultsFolder + "rand{}/".format(i)
            self.runManager(resultsFolder, labelNameCurrentRand)

    def runManager(self, resultsFolder, labelNameCurrentRand):
        dataFolder = "Data/WT/"
        plantNames = ["P1", "P2", "P5", "P6", "P8", "P9", "P10", "P11"]
        testPlants = ["P2", "P9"]
        if self.divEventPred:
            modelType =  {"modelType":"svm","kernel":"linear"}
        else:
            modelType =  {"modelType":"svm","kernel":"rbf"}
        usePreviousTrainedModelsIfPossible = False
        onlyTestModelWithoutTrainingData = False
        useManualCentres = True
        # print options:
        printBalancedLabelCount = True
        nSplits = "per plant"
        centralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                            "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                            "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                            "P6":[[861], [], [], [2109, 2176], [2381]],
                            "P8":[[3241, 2869, 3044], [3421, 3657], [], [], [358, 189]],
                            "P9":[[1047, 721, 1048], [7303, 7533], [6735, 7129], [2160, 2228], [7366, 7236]],
                            "P10":[[1511, 1524], [7281, 7516, 7534], [], [7634, 7722, 7795, 7794], [1073, 1074, 892]],
                            "P11":[[1751], [9489], [9759, 9793], [3300, 3211, 3060], [3956, 3979]]}
        doHyperParameterisation = False
        normalisePerTissue = False
        normaliseTrainTestData = True
        normaliseTrainValTestData = False
        featureProperty = "combinedTable"
        runModelTraining = True
        manager = PredictonManager(plantNames=plantNames,
                               testPlants=testPlants,
                               featureProperty=featureProperty,
                               dataFolder=dataFolder,
                               featureAndLabelFolder=self.setFeatureAndLabelFolder,
                               givenFeatureName=self.givenFeatureName,
                               resultsFolder=resultsFolder,
                               nSplits=nSplits,
                               labelName=labelNameCurrentRand,
                               modelType=modelType,
                               runModelTraining = runModelTraining,
                               runModelTesting=self.includeTesting,
                               excludeDividingNeighbours=self.excludeDividingNeighbours,
                               centralCellsDict=centralCellsDict,
                               normalisePerTissue=normalisePerTissue,
                               normaliseTrainTestData=normaliseTrainTestData,
                               normaliseTrainValTestData=normaliseTrainValTestData,
                               doHyperParameterisation=doHyperParameterisation,
                               hyperParameters=self.hyperParameters,
                               useGivenHyperParForTesting=True,
                               folderToSaveVal=resultsFolder,
                               setFeatureAndLabelFolder=self.setFeatureAndLabelFolder,
                               useOnlyTwo=self.useOnlyTwo)

    def combineRandomResults(self, nrOfRandomRuns, basicResultsFolder):
        print("Combining Results")
        if self.includeTesting:
            testingTxt = "WithTesting"
        else:
            testingTxt = ""
        allResultsTables = []
        if self.excludeDividingNeighbours:
            excludedValue = 1
            modelFolderName = "svm_k1_combinedTable_l3f0n1c0bal0ex1/"
        else:
            excludedValue = 0
            if self.divEventPred:
                modelFolderName = "svm_k1_combinedTable_l3f0n1c0bal0ex0/"
            else:
                modelFolderName = "svm_k2_combinedTable_l3f0n1c0bal0ex0/"
        for i in range(1, nrOfRandomRuns+1):
            resultsFolder = basicResultsFolder + "rand{}/".format(i) + modelFolderName
            resultsTable = pd.read_csv(resultsFolder + "results{}.csv".format(testingTxt), index_col=0)
            allResultsTables.append(resultsTable.to_numpy())
        allResultsTables = np.asarray(allResultsTables)
        meanResultsTable = np.mean(allResultsTables, axis=0)
        resultsTable.iloc[:, :] = meanResultsTable
        meanResultsTableName = basicResultsFolder + "combinedResults{}Of_{}_randomizedRuns_ex{}.csv".format(testingTxt, nrOfRandomRuns, excludedValue)
        if self.printEndResult:
            print(str(resultsTable))
        resultsTable.to_csv(meanResultsTableName)

def mainCallDivPredRandomization():
    for featureSet in ["area", "allTopos", "topoAndBio"]:
        print("featureSet", featureSet)
        myRandomLabelPredictior = RandomLabelPredictior(nrOfRandomRuns=100,
                              recreateRadomLabels=True, runModel=True,
                              featureSet=featureSet,
                              divEventPred=True, includeTesting=True,
                              excludeDividingNeighbours=False)

def mainCallTopoPredRandomization(excludeDividingNeighboursPar=[True], # [True, False]
                                  givenSets= ["bio", "allTopos", "topoAndBio"]):
    for excludeDividingNeighbours in excludeDividingNeighboursPar:
        for featureSet in givenSets:
            print("featureSet", featureSet, "excludeDividingNeighbours", excludeDividingNeighbours)
            myRandomLabelPredictior = RandomLabelPredictior(nrOfRandomRuns=100,
                                  recreateRadomLabels=True, featureSet=featureSet,
                                  excludeDividingNeighbours=excludeDividingNeighbours,
                                  divEventPred=False, includeTesting=True)

def main():
    # mainCallDivPredRandomization()
    mainCallTopoPredRandomization()

if __name__ == '__main__':
    main()
