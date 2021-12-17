import copy
import numpy as np
import pickle
import sys
sys.path.insert(0, "./Code/Classifiers/")

from BalancedXFold import BalancedXFold
from DividingCellInTableIdentifier import DividingCellInTableIdentifier
from ModelEnsambleUtiliser import ModelEnsambleUtiliser
from MyScorer import MyScorer
from NestedModelCreator import NestedModelCreator
from pathlib import Path
from sklearn.model_selection import KFold

class PredictonModelCreator (object):

    isCellTest=None
    verbosity=0

    def __init__(self, features, labels, testPlants, modelType, nSplits=5,
                 seed=42, folderToSave=None,
                 normaliseTrainTestData=False,
                 normaliseTrainValTestData=False,
                 excludeDividingNeighboursDict=False,
                 doHyperParameterisation=False,
                 hyperParameterRange=None,
                 hyperParameters=None,
                 nestedModelProp=False,
                 allNestedModelProp=[[0, [1,2]], [1, [0,2]], [2, [0,1]]],
                 modelEnsambleNumber=False,
                 balanceData=True,
                 parametersToAddOrOverwrite=None,
                 folderToSaveVal=None,
                 useOnlyTwo=True,
                 printCurrentSplit=False, printLabelsAndCount=False, printModelRuntime=False):
        self.features = features
        self.labels = labels
        self.testPlants = testPlants
        self.modelType = modelType
        self.folderToSave = folderToSave
        self.normaliseTrainTestData = normaliseTrainTestData
        self.normaliseTrainValTestData = normaliseTrainValTestData
        self.excludeDividingNeighboursDict = excludeDividingNeighboursDict
        self.seed = seed
        self.nSplits = nSplits
        self.doHyperParameterisation = doHyperParameterisation
        self.hyperParameterRange = hyperParameterRange
        self.hyperParameters = hyperParameters
        self.nestedModelProp = nestedModelProp
        self.allNestedModelProp = allNestedModelProp
        self.modelEnsambleNumber = modelEnsambleNumber
        self.balanceData = balanceData
        self.parametersToAddOrOverwrite = parametersToAddOrOverwrite
        self.folderToSaveVal = folderToSaveVal
        self.useOnlyTwo = useOnlyTwo
        self.nrOfClasses = 2 if self.useOnlyTwo is True else 3
        self.printCurrentSplit = printCurrentSplit
        self.printLabelsAndCount = printLabelsAndCount
        self.printModelRuntime = printModelRuntime
        np.random.seed(self.seed)
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.allTrainPerfromance = None
        self.allValidationPerfromance = None

    def TrainAndTestModel(self, runModelTraining=True, printSampleOverview=False,
                          usePreviouslyTrainedModels=None, usePreviousTrainedModelsIfPossible=True,
                          normaliseOnTestData=False):
        self.runModelTraining = runModelTraining
        self.usePreviouslyTrainedModels = usePreviouslyTrainedModels
        self.X_train, self.y_train, self.X_test, self.y_test = self.trainTestSplit(self.testPlants)
        if printSampleOverview:
            print("sample overview:")
            print("nr of train", len(self.X_train), "nr of test", len(self.X_test))
            print("train labels and counts",np.unique(self.y_train, return_counts=True))
            print("test labels and counts", np.unique(self.y_test, return_counts=True))
            print("nr of features used:",self.X_train.shape[1])
        if normaliseOnTestData:
            self.X_test = self.doZNormalise(self.X_test)
        elif self.normaliseTrainTestData:
            self.X_train, normParameters = self.doZNormalise(self.X_train, returnParameter=True)
            self.X_test = self.doZNormalise(self.X_test, useParameters=normParameters)
        elif self.normaliseTrainValTestData:
            normParameters = [np.mean(self.X_train, axis=0), np.std(self.X_train, axis=0)]
            self.X_test = self.doZNormalise(self.X_test, useParameters=normParameters)
        if self.runModelTraining:
            import time
            startTime = time.time()
            self.trainModel(self.X_train, self.y_train, nSplits=self.nSplits, seed=self.seed,
                            usePreviousTrainedModelsIfPossible=usePreviousTrainedModelsIfPossible)
            endTime = time.time()-startTime
            if self.printModelRuntime:
                print("actualTime:", endTime)

    def trainTestSplit(self, testPlants):
        plantOfCellInLabels = self.labels.iloc[:, 0]
        self.isCellTest = np.isin(plantOfCellInLabels, testPlants)
        y_train, y_test = self.determineTrainTestLabels(self.isCellTest)
        X_train, X_test = self.determineTrainTestFeatures(self.isCellTest)
        return X_train, y_train, X_test, y_test

    def determineTrainTestLabels(self, isCellTest):
        y_test = self.labels.iloc[isCellTest, -1].to_numpy()
        y_train = self.labels.iloc[np.invert(isCellTest), -1].to_numpy()
        return y_train, y_test

    def determineTrainTestFeatures(self, isCellTest):
        if self.useOnlyTwo:
            tillProperty = 3
        else:
            tillProperty = 4
        x_Data = self.features.iloc[:, tillProperty:].to_numpy()
        # assert not np.any(np.isnan(x_Data)), "contains na"
        if np.any(np.nan==x_Data):
            columnsContainingZero = np.unique(np.where(np.nan==x_Data)[1])
            print("column contains nan is removed: {}".format(columnsContainingZero))
            x_Data = np.delete(x_Data, columnsContainingZero, axis=1)
        assert not np.any(np.inf==x_Data), "contains inf"
        X_train = x_Data[np.invert(isCellTest), :]
        X_test = x_Data[isCellTest, :]
        return X_train, X_test

    def findCorrespondingFeatureRow(self, labelIdx, columnsNeedingToFit,
                                    correspondingFeatureColumns):
        assert len(columnsNeedingToFit) == len(correspondingFeatureColumns), "The columnsNeedingToFit and the correspondingFeatureColumns need to have the same length. {} != {}".format(len(columnsNeedingToFit), len(correspondingFeatureColumns))
        valuesToFit = self.labels.iloc[labelIdx, columnsNeedingToFit]
        selectedFeatureColumns = self.features.iloc[:, correspondingFeatureColumns]
        isRowSelected = selectedFeatureColumns.to_numpy() == valuesToFit.to_numpy()
        isRowSelected = np.sum(isRowSelected, axis=1)
        isRowSelected = isRowSelected == len(columnsNeedingToFit)
        assert np.any(isRowSelected), "No feature source was corresponding to this label idx {} properties {}".format(labelIdx, valuesToFit)
        selectedIdx = np.where(isRowSelected)[0]
        assert len(selectedIdx) == 1, "For the cell (from label idx {}), more than one idx in the feature set is found to match {}. {} != 1".format(labelIdx, selectedIdx, len(selectedIdx))
        return selectedIdx[0]

    def doZNormalise(self, X_train, useParameters=None, returnParameter=False):
        if not useParameters is None:
            mean = useParameters[0]
            std = useParameters[1]
        else:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
        X_train = (X_train-mean)/std
        if returnParameter:
            return [X_train, [mean, std]]
        else:
            return X_train

    def trainModel(self, X_train, y_train, nSplits=5, seed=42,
                   usePreviousTrainedModelsIfPossible=True):
        self.cvModels, self.alreadyRunEncapsulatedModels = [], []
        self.nrOfPreviouslyDOneCVModles = 0
        self.currentSplit = 1
        self.allTrainPerfromance, self.allValidationPerfromance = [], []
        self.valXs, self.valYs = [], []
        if self.usePreviouslyTrainedModels is None:
            self.modelFilename = self.folderToSaveVal + "normalFittedModelEncaps.pkl"
        else:
            self.modelFilename = self.folderToSaveVal + "ReFittedModelEncaps.pkl"
        if not self.folderToSaveVal is None and usePreviousTrainedModelsIfPossible is True:
            if Path(self.modelFilename).is_file():
                self.alreadyRunEncapsulatedModels = pickle.load(open(self.modelFilename, "rb"))
                self.nrOfPreviouslyDOneCVModles = len(self.alreadyRunEncapsulatedModels)
        if type(nSplits) == int:
            self.trainBasicCrossValSplit(X_train, y_train, nSplits, seed)
        else:
            self.trainPerPlantSplit(X_train, y_train)
        if not self.folderToSaveVal is None:
            Path(self.folderToSaveVal).mkdir(parents=True, exist_ok=True)
            pickle.dump(self.valXs, open(self.folderToSaveVal+"valXs.pkl", "wb"))
            pickle.dump(self.valYs, open(self.folderToSaveVal+"valYs.pkl", "wb"))

    def trainBasicCrossValSplit(self, X_train, y_train, nSplits, seed):
        xFold = BalancedXFold(n_splits=nSplits, random_state=seed)
        for trainIdx, validationIdx in xFold.split(X_train, y_train):
            if self.printCurrentSplit or True:
                print("split {}/{}".format(self.currentSplit, self.nSplits))
            trainPerformance, validationPerformance = self.trainAndValidateModel(X_train, y_train, trainIdx, validationIdx)
            self.allTrainPerfromance.append(trainPerformance)
            self.allValidationPerfromance.append(validationPerformance)
            self.currentSplit += 1

    def trainPerPlantSplit(self, X_train, y_train, plantNameColIdx=0):
        plantNameOfTrainData = self.GetPlantNameOfTrainingData(plantNameColIdx)
        _, uniqueIdx = np.unique(plantNameOfTrainData, return_index=True)
        uniquePlantNames = plantNameOfTrainData[uniqueIdx]
        numberOfPlants = len(uniquePlantNames)
        for currentValidationPlant in uniquePlantNames:
            if self.printCurrentSplit or True:
                print("val plant {} with split {}/{}".format(currentValidationPlant, self.currentSplit, numberOfPlants))
            isValidationPlant = np.isin(plantNameOfTrainData, currentValidationPlant)
            validationIdx, trainIdx = np.where(isValidationPlant)[0], np.where(np.invert(isValidationPlant))[0]
            trainPerformance, validationPerformance = self.trainAndValidateModel(X_train, y_train, trainIdx, validationIdx)
            self.allTrainPerfromance.append(trainPerformance)
            self.allValidationPerfromance.append(validationPerformance)
            self.currentSplit += 1

    def trainAndValidateModel(self, X, y, trainIdx, validationIdx):
        currentX_Train, validationX = X[trainIdx, :], X[validationIdx, :]
        currentY_Train, validationY = y[trainIdx], y[validationIdx]
        if not self.normaliseTrainTestData and self.normaliseTrainValTestData:
            currentX_Train, normParameters = self.doZNormalise(currentX_Train, returnParameter=True)
            validationX = self.doZNormalise(validationX, useParameters=normParameters)
        if self.printLabelsAndCount:
            print("train", np.unique(currentY_Train, return_counts=True))
            print("val", np.unique(validationY, return_counts=True))
        if self.balanceData:
            currentX_Train, currentY_Train = self.balanceFeaturesAndLabels(currentX_Train, currentY_Train)
            validationX, validationY = self.balanceFeaturesAndLabels(validationX, validationY)
        self.valXs.append(validationX)
        self.valYs.append(validationY)
        if self.usePreviouslyTrainedModels is None:
            if self.currentSplit > self.nrOfPreviouslyDOneCVModles:
                if self.verbosity > 0:
                    print("train model {}".format(self.currentSplit))
                model = self.trainModelWith(currentX_Train, currentY_Train, validationX, validationY)
            else:
                model = self.alreadyRunEncapsulatedModels[self.currentSplit-1]
        else:
            if self.verbosity > 0:
                print("refitting previous trained model {}".format(self.currentSplit))
            model = self.usePreviouslyTrainedModels[self.currentSplit-1].best_estimator_
            model.fit(currentX_Train, currentY_Train)
        self.cvModels.append(model.GetModel())
        trainPerformance = model.TestModel(currentX_Train, currentY_Train)
        notUsedEnsamble = (self.modelEnsambleNumber == 1 or self.modelEnsambleNumber == False) and not (self.nestedModelProp is True)
        if notUsedEnsamble is False:
            model.PrintNrOfTiesBetweenCount("from train data")
        validationPerformance = model.TestModel(validationX, validationY)
        if notUsedEnsamble is False:
            model.PrintNrOfTiesBetweenCount("from val data")
        return trainPerformance, validationPerformance

    def balanceFeaturesAndLabels(self, usedX_train, usedY_train):
        label, counts = np.unique(usedY_train, return_counts=True)
        if self.useOnlyTwo:
            assert len(counts) == 2, "The number of labels needs to be two. The labels {} are present. {} != 2".format(label, len(counts))
            if counts[0] == counts[1]:
                return usedX_train, usedY_train
            nrOfLabels = 2
        else:
            assert len(counts) == 3, "The number of labels needs to be three. The labels {} are present. {} != 3".format(label, len(counts))
            if counts[0] == counts[1] == counts[2]:
                return usedX_train, usedY_train
            nrOfLabels = 3
        minCount = np.min(counts)
        selectedIdx = np.zeros(minCount*nrOfLabels, dtype=int)
        nrOfSelectedLabels = np.zeros(nrOfLabels)
        j = 0
        for i in range(len(usedY_train)):
            currentLabel = usedY_train[i]
            if  nrOfSelectedLabels[currentLabel] < minCount:
                nrOfSelectedLabels[currentLabel] += 1
                selectedIdx[j] = i
                j += 1
        usedY_train = copy.deepcopy(usedY_train[selectedIdx])
        usedX_train = copy.deepcopy(usedX_train[selectedIdx])
        return usedX_train, usedY_train

    def trainModelWith(self, currentX_Train, currentY_Train, validationX, validationY):
        model = NestedModelCreator(currentX_Train, currentY_Train,
                                  validationX, validationY, modelType=self.modelType,
                                  performanceModus="all performances",
                                  doHyperParameterisation=self.doHyperParameterisation,
                                  hyperParameterRange=self.hyperParameterRange,
                                  hyperParameters=self.hyperParameters,
                                  parametersToAddOrOverwrite=self.parametersToAddOrOverwrite,
                                  nestedModelProp=self.nestedModelProp,
                                  nrOfClasses=self.nrOfClasses)
        if not self.folderToSaveVal is None:
            self.alreadyRunEncapsulatedModels.append(model)
            pickle.dump(self.alreadyRunEncapsulatedModels, open(self.modelFilename, "wb"))
        return model

    def testModelOn(self, X, y):
        pass

    def GetPlantNameOfTrainingData(self, plantNameColIdx=0):
        assert not self.features is None, "The features are not yet defined."
        assert not self.isCellTest is None, "isCellTest array are not yet defined."
        return self.features.iloc[np.invert(self.isCellTest), plantNameColIdx].to_numpy()

    def GetTrainAndTestUniqueTissueIdentifiers(self, plantNameColIdx=0, timePointColIdx=1):
        trainTissueIds = self.GetTissueId(fromTest=False, plantNameColIdx=plantNameColIdx, timePointColIdx=timePointColIdx)
        testTissueIds = self.GetTissueId(fromTest=True, plantNameColIdx=plantNameColIdx, timePointColIdx=timePointColIdx)
        return trainTissueIds, testTissueIds

    def GetTissueId(self, fromTest=False, plantNameColIdx=0, timePointColIdx=1):
        columnNamesToGroupBy = list(np.asarray(self.labels.columns)[[plantNameColIdx, timePointColIdx]])
        if not fromTest is None:
            if fromTest:
                labelDf = self.labels.iloc[self.isCellTest, :].copy()
            else:
                labelDf = self.labels.iloc[np.invert(self.isCellTest), :].copy()
        else:
            labelDf = self.labels.copy()
        labelDf.set_index(np.arange(len(labelDf)), inplace=True)
        perTissueGroupedDfs = labelDf.groupby(columnNamesToGroupBy, axis=0)
        tissueIds = self.GetTissueIdFromGrouping(perTissueGroupedDfs)
        return tissueIds

    def GetTissueIdFromGrouping(self, perTissueGroupedDfs):
        numberOfSamples = 0
        for i, grouping in enumerate(perTissueGroupedDfs):
            numberOfSamples += len(grouping[1])
        tissueId = np.full(numberOfSamples, np.NaN)
        for i, grouping in enumerate(perTissueGroupedDfs):
            idx = grouping[1].index
            tissueId[idx] = i
        return tissueId

    def GetTrainPerformance(self):
        return self.allTrainPerfromance

    def GetValidationPerformance(self):
        return self.allValidationPerfromance

    def GetTrainTestDataAndLabels(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def GetCVModels(self):
        return self.cvModels

    def GetNamesOfUsedFeatures(self):
        return self.namesOfUsedFeatures

    def GetIsCellTest(self):
        return self.isCellTest

def main():
    import pandas as pd
    from pathlib import Path
    dataFolder = "Data/WT/topoPredData/diff/manualCentres/"
    resultsFolder = "Results/Tissue Mapping/"
    featureProperty = ["topology", "topologyArea", "topologyWall", "topologyDist", "combinedTables"]
    allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist"]
    normalisationProperties = [False, True]
    doHyperParameterisation = True
    labelFilenameProperties = [dataFolder + "combinedLabels.csv"]
    allResults = []
    plantNames = ["P1", "P2", "P5", "P6", "P8"]
    testPlants = ["P2"]
    modelType = {"svm":{"kernel":"rbf"}}
    excludeDividingNeighboursDict = {'P1': {0: 'Data/WT/P1/parentLabelingP1T0T1.csv', 1: 'Data/WT/P1/parentLabelingP1T1T2.csv', 2: 'Data/WT/P1/parentLabelingP1T2T3.csv', 3: 'Data/WT/P1/parentLabelingP1T3T4.csv'},
                                'P2': {0: 'Data/WT/P2/parentLabelingP2T0T1.csv', 1: 'Data/WT/P2/parentLabelingP2T1T2.csv', 2: 'Data/WT/P2/parentLabelingP2T2T3.csv', 3: 'Data/WT/P2/parentLabelingP2T3T4.csv'},
                                'P5': {0: 'Data/WT/P5/parentLabelingP5T0T1.csv', 1: 'Data/WT/P5/parentLabelingP5T1T2.csv', 2: 'Data/WT/P5/parentLabelingP5T2T3.csv', 3: 'Data/WT/P5/parentLabelingP5T3T4.csv'},
                                'P6': {0: 'Data/WT/P6/parentLabelingP6T0T1.csv', 1: 'Data/WT/P6/parentLabelingP6T1T2.csv', 2: 'Data/WT/P6/parentLabelingP6T2T3.csv', 3: 'Data/WT/P6/parentLabelingP6T3T4.csv'},
                                'P8': {0: 'Data/WT/P8/parentLabelingP8T0T1.csv', 1: 'Data/WT/P8/parentLabelingP8T1T2.csv', 2: 'Data/WT/P8/parentLabelingP8T2T3.csv', 3: 'Data/WT/P8/parentLabelingP8T3T4.csv'}}
    for labelFilename in labelFilenameProperties:
        results = []
        for selectedFeatureTable in featureProperty:
            trainPerfromance = []
            validationPerfromance = []
            for normaliseTrainTestData in normalisationProperties:
                print(selectedFeatureTable)
                if selectedFeatureTable == "combinedTables":
                    featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(allFeatureProperties[0])
                    features = pd.read_csv(featureFilename)
                    for property in allFeatureProperties[1:]:
                        featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(property)
                        table2 = pd.read_csv(featureFilename)
                        features = pd.concat([features, table2.iloc[:, 3:]], axis=1, ignore_index=True)
                else:
                    featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(selectedFeatureTable)
                    features = pd.read_csv(featureFilename,sep=",")
                labels = pd.read_csv(labelFilename,sep=",")
                folderToSave = resultsFolder+selectedFeatureTable+"/"
                folderToSave = None
                if not folderToSave is None:
                    Path(folderToSave).mkdir(parents=True, exist_ok=True)
                modelCreator = PredictonModelCreator(features, labels,
                                                             testPlants, modelType=modelType,
                                                             folderToSave=folderToSave,
                                                             normaliseTrainTestData=normaliseTrainTestData,
                                                             excludeDividingNeighboursDict=excludeDividingNeighboursDict,
                                                             doHyperParameterisation=doHyperParameterisation)
                modelCreator.TrainAndTestModel()
                trainP = modelCreator.GetTrainPerformance()
                validationP = modelCreator.GetValidationPerformance()
                print("train val performance with hyper", trainP, validationP)
                if selectedFeatureTable == "combinedTables":
                    featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(allFeatureProperties[0])
                    features = pd.read_csv(featureFilename)
                    for property in allFeatureProperties[1:]:
                        featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(property)
                        table2 = pd.read_csv(featureFilename)
                        features = pd.concat([features, table2.iloc[:, 3:]], axis=1, ignore_index=True)
                else:
                    featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(selectedFeatureTable)
                    features = pd.read_csv(featureFilename,sep=",")
                labels = pd.read_csv(labelFilename,sep=",")
                modelCreator = PredictonModelCreator(features, labels,
                                                             testPlants, modelType=modelType,
                                                             normaliseTrainTestData=normaliseTrainTestData,
                                                             excludeDividingNeighboursDict=excludeDividingNeighboursDict,
                                                             doHyperParameterisation=False)
                modelCreator.TrainAndTestModel()
                trainP = modelCreator.GetTrainPerformance()
                validationP = modelCreator.GetValidationPerformance()
                print("train val performance without hyper", trainP, validationP)
                sys.exit()
                trainPerfromance.append(trainP)
                validationPerfromance.append(validationP)
            currentResults = [selectedFeatureTable, "train", np.round(np.mean(trainPerfromance[0]), 1), np.round(np.mean(trainPerfromance[1]), 1)]
            currentResults.extend(list(trainPerfromance[0]))
            currentResults.extend(list(trainPerfromance[1]))
            results.append(currentResults)
            currentResults = [selectedFeatureTable, "val", np.round(np.mean(validationPerfromance[0]), 1), np.round(np.mean(validationPerfromance[1]), 1)]
            currentResults.extend(list(validationPerfromance[0]))
            currentResults.extend(list(validationPerfromance[1]))
            results.append(currentResults)
        results = np.vstack(results)
        allResults.append(results)
    allResults = np.hstack(allResults)
    allResults = pd.DataFrame(allResults)
    print(allResults)
    allResults.to_csv(resultsFolder+"2label_noDividingNeighbours_ratio_modelResults.csv", index=False)

if __name__ == '__main__':
    main()
