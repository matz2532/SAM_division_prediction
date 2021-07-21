import numpy as np
import pandas as pd
import sys

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
            self.runModels(nrOfRandomRuns, self.resultsFolder)
        self.combineRandomResults(nrOfRandomRuns, self.resultsFolder)

    def setBaseParameters(self, featureSet):
        if self.divEventPred is True:
            insert = "divEventData/"
            self.useOnlyTwo = True
        else:
            insert = "topoPredData/diff/"
            self.useOnlyTwo = False
        assert featureSet in ["area", "bio", "topology", "topoAndBio", "allTopos"], "featureSet needs to be area, bio, topology, allTopos, or topoAndBio, featureSet is {}".format(featureSet)
        self.setFeatureAndLabelFolder = "Data/WT/{}manualCentres/{}/".format(insert, featureSet)
        self.resultsFolder = "Results/{}manualCentres/{}/".format(insert, featureSet)
        self.givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(featureSet)
        self.basicLabelName = "combinedLabels{}.csv"
        self.labelName = "combinedLabels.csv"

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
            if i % 100 == 0:
                print("currentRandomSample: {}/{}".format(i, nrOfRandomRuns))
            labelNameCurrentRand = self.basicLabelName.format(i)
            resultsFolder = basicResultsFolder + "rand{}/".format(i)
            self.runManager(resultsFolder, labelNameCurrentRand)

    def runManager(self, resultsFolder, labelNameCurrentRand):
        plantNames = ["P1", "P2", "P5", "P6", "P8"]
        testPlants = ["P2"]
        allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist", "combinedTable"]#, "combinedTable"]
        featureProperty = allFeatureProperties[-1]
        dataFolder = "Data/WT/"
        featureAndLabelFolder = "Data/WT/Tissue mapping/"
        runModelTraining = True
        centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                            "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                            "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                            "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                            "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
        normalisePerTissue = False
        normaliseTrainTestData = True
        normaliseTrainValTestData = False
        doHyperParameterisation = False
        hyperParameters = None
        nestedModelProp = False
        modelEnsambleNumber = 1
        selectedData = 1
        parametersToAddOrOverwrite = None
        parName = None
        modelNameExtension = ""
        modelType = {"modelType":"svm", "kernel":"rbf"}
        folderToSaveVal = resultsFolder
        if self.printResultsFolder:
            print("resultsFolder: " + resultsFolder)
        manager = PredictonManager(plantNames=plantNames,
                               testPlants=testPlants,
                               featureProperty=featureProperty,
                               dataFolder=dataFolder,
                               featureAndLabelFolder=featureAndLabelFolder,
                               givenFeatureName=self.givenFeatureName,
                               resultsFolder=resultsFolder,
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
                               hyperParameters=hyperParameters,
                               nestedModelProp=nestedModelProp,
                               modelEnsambleNumber=modelEnsambleNumber,
                               selectedData=selectedData,
                               parametersToAddOrOverwrite=parametersToAddOrOverwrite,
                               specialParName=parName,
                               modelNameExtension=modelNameExtension,
                               folderToSaveVal=folderToSaveVal,
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
            modelFolderName = "svm_k2_combinedTable_l3f0n1c0ex1/"
        else:
            excludedValue = 0
            modelFolderName = "svm_k2_combinedTable_l3f0n1c0ex0/"
        for i in range(1, nrOfRandomRuns+1):
            resultsFolder = basicResultsFolder + "rand{}/".format(i) + modelFolderName
            resultsTable = pd.read_csv(resultsFolder + "results{}.csv".format(testingTxt), index_col=0)
            allResultsTables.append(resultsTable.to_numpy())
        allResultsTables = np.asarray(allResultsTables)
        meanResultsTable = np.mean(allResultsTables, axis=0)
        std = [np.std(allResultsTables[:, :-2-self.includeTesting, i]) for i in range(meanResultsTable.shape[1])]
        meanResultsTable[-1-self.includeTesting, :] = std
        resultsTable.iloc[:, :] = meanResultsTable
        if self.includeTesting:
            stdTest = [np.std(allResultsTables[:, -1, i]) for i in range(meanResultsTable.shape[1])]
            stdTest = pd.DataFrame(stdTest, columns=["std testing"], index=resultsTable.columns)
            stdTest = stdTest.T
            resultsTable = pd.concat([resultsTable, stdTest], axis=0)
        meanResultsTableName = basicResultsFolder + "combinedResults{}Of_{}_randomizedRuns_ex{}.csv".format(testingTxt, nrOfRandomRuns, excludedValue)
        if self.printEndResult:
            print(str(resultsTable))
        resultsTable.to_csv(meanResultsTableName)

def mainCallDivPredRandomization():
    for featureSet in ["area", "allTopos", "topology", "topoAndBio"]:
        print("featureSet", featureSet)
        myRandomLabelPredictior = RandomLabelPredictior(nrOfRandomRuns=1000,
                              recreateRadomLabels=False, featureSet=featureSet,
                              divEventPred=True, includeTesting=True)

def mainCallTopoPredRandomization(excludeDividingNeighboursPar=[False], # [True, False]
                                  featureSetPar=["topology", "topoAndBio"] # ["bio", "allTopos", "topology", "topoAndBio"]
                                  ):
    for excludeDividingNeighbours in excludeDividingNeighboursPar:
        for featureSet in featureSetPar:
            print("featureSet", featureSet, "excludeDividingNeighbours", excludeDividingNeighbours)
            myRandomLabelPredictior = RandomLabelPredictior(nrOfRandomRuns=1000,
                                  recreateRadomLabels=True, featureSet=featureSet,
                                  excludeDividingNeighbours=excludeDividingNeighbours,
                                  divEventPred=False, includeTesting=True)

def main():
    # mainCallDivPredRandomization()
    mainCallTopoPredRandomization()

if __name__ == '__main__':
    main()
