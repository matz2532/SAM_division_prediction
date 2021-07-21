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
                 selectedData=1,
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
        self.selectedData = selectedData
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
        columnsNeedingToFit = np.arange(tillProperty-1)
        correspondingFeatureColumns = np.arange(tillProperty-1)
        labelCellCorespondingToFeatureIdx = np.zeros(len(isCellTest), dtype=int)
        for i in range(len(isCellTest)):
            idx = self.findCorrespondingFeatureRow(i, columnsNeedingToFit,
                                                   correspondingFeatureColumns)
            labelCellCorespondingToFeatureIdx[i] = idx
        selectedTrainCells = labelCellCorespondingToFeatureIdx[np.invert(isCellTest)]
        selectedTestCells = labelCellCorespondingToFeatureIdx[isCellTest]
        X_train = self.features.iloc[selectedTrainCells, tillProperty:].to_numpy()
        X_test = self.features.iloc[selectedTestCells, tillProperty:].to_numpy()
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
        if self.modelEnsambleNumber == 1 or self.modelEnsambleNumber == False:
            xFold = BalancedXFold(n_splits=nSplits, random_state=seed)
        else:
            xFold = KFold(n_splits=nSplits, shuffle=True, random_state=seed)
        self.cvModels = []
        nrOfPreviouslyDOneCVModles = 0
        alreadyRunEncapsulatedModels = []
        if self.usePreviouslyTrainedModels is None:
            modelFilename = self.folderToSaveVal + "normalFittedModelEncaps.pkl"
        else:
            modelFilename = self.folderToSaveVal + "ReFittedModelEncaps.pkl"
        if not self.folderToSaveVal is None and usePreviousTrainedModelsIfPossible is True:
            if Path(modelFilename).is_file():
                alreadyRunEncapsulatedModels = pickle.load(open(modelFilename, "rb"))
                nrOfPreviouslyDOneCVModles = len(alreadyRunEncapsulatedModels)
        allTrainPerfromance = []
        allValidationPerfromance = []
        currentSplit = 1
        if self.useOnlyTwo:
            avgConfMt = np.zeros((3,3))
        else:
            avgConfMt = np.zeros((2,2))
        valXs, valYs = [], []
        for trainIdx, validationIdx in xFold.split(X_train, y_train):
            if self.printCurrentSplit or True:
                print("split {}/{}".format(currentSplit, nSplits))
            currentX_Train = X_train[trainIdx, :]
            currenty_Train = y_train[trainIdx]
            validationX = X_train[validationIdx, :]
            validationy = y_train[validationIdx]
            if not self.normaliseTrainTestData and self.normaliseTrainValTestData:
                currentX_Train, normParameters = self.doZNormalise(currentX_Train, returnParameter=True)
                validationX = self.doZNormalise(validationX, useParameters=normParameters)
            if self.printLabelsAndCount:
                print("train", np.unique(currenty_Train, return_counts=True))
                print("val", np.unique(validationy, return_counts=True))
            validationX, validationy = self.balanceData(validationX, validationy)
            valXs.append(validationX)
            valYs.append(validationy)
            if self.usePreviouslyTrainedModels is None:
                print("fitting model {}".format(currentSplit))
                notUsedEnsamble = (self.modelEnsambleNumber == 1 or self.modelEnsambleNumber == False) and not (self.nestedModelProp is True)
                if currentSplit > nrOfPreviouslyDOneCVModles:
                    model = self.calcModelForEnsamble(currentX_Train, currenty_Train,
                                                      validationX, validationy,
                                                      selectedData=self.selectedData)
                    if self.doHyperParameterisation and not self.folderToSave is None:
                        allHyperParameters = myModelEnsambleUtiliser.GetAllHyperParameters()
                        filename = self.folderToSave + "hyperParameters_run{}.pkl".format(currentSplit)
                        pickle.dump(allHyperParameters, open(filename, "wb"))
                    self.cvModels.append(model.GetModel())
                    if not self.folderToSaveVal is None:
                        alreadyRunEncapsulatedModels.append(model)
                        pickle.dump(alreadyRunEncapsulatedModels, open(modelFilename, "wb"))
                else:
                    model = alreadyRunEncapsulatedModels[currentSplit-1]
                trainPerformance = model.TestModel(currentX_Train, currenty_Train)
                if notUsedEnsamble is False:
                    model.PrintNrOfTiesBetweenCount("from train data")
                validationPerformance = model.TestModel(validationX, validationy)
                if notUsedEnsamble is False:
                    model.PrintNrOfTiesBetweenCount("from val data")
            else:
                print("refitting previous trained model {}".format(currentSplit))
                model = self.usePreviouslyTrainedModels[currentSplit-1].best_estimator_
                model.fit(currentX_Train, currenty_Train)
            y_pred = model.predict(validationX)
            allTrainPerfromance.append(trainPerformance)
            allValidationPerfromance.append(validationPerformance)
            currentSplit += 1
        if not self.folderToSaveVal is None:
            Path(self.folderToSaveVal).mkdir(parents=True, exist_ok=True)
            pickle.dump(valXs, open(self.folderToSaveVal+"valXs.pkl", "wb"))
            pickle.dump(valYs, open(self.folderToSaveVal+"valYs.pkl", "wb"))
        self.allTrainPerfromance = allTrainPerfromance
        self.allValidationPerfromance = allValidationPerfromance

    def calcModelForEnsamble(self, currentX_Train, currenty_Train,
                             validationX, validationy, selectedData=0.8):
        modelList = []
        nrOfTrainSamples = len(currenty_Train)
        sampleIdx = np.arange(nrOfTrainSamples)
        nrOfSelectedSamples = int(nrOfTrainSamples*selectedData)
        for modelNr in range(self.modelEnsambleNumber):
            print("model {}/{}".format(modelNr+1, self.modelEnsambleNumber))
            np.random.shuffle(sampleIdx)
            selectedSampleIndices = sampleIdx[:nrOfSelectedSamples]
            selectedX_train = currentX_Train[selectedSampleIndices]
            selectedy_train = currenty_Train[selectedSampleIndices]
            selectedX_train, selectedy_train = self.balanceData(selectedX_train, selectedy_train)
            if self.nestedModelProp is True:
                for nestedModelProp in self.allNestedModelProp:
                    print("current nestedModelProp", nestedModelProp)
                    model = NestedModelCreator(selectedX_train, selectedy_train,
                                               validationX, validationy, modelType=self.modelType,
                                               performanceModus="all performances", # "accuracy",
                                               doHyperParameterisation=self.doHyperParameterisation,
                                               hyperParameterRange=self.hyperParameterRange,
                                               hyperParameters=self.hyperParameters,
                                               parametersToAddOrOverwrite=self.parametersToAddOrOverwrite,
                                               nestedModelProp=nestedModelProp,
                                               nrOfClasses=self.nrOfClasses)
                    modelList.append(model)
            else:
                model = NestedModelCreator(selectedX_train, selectedy_train,
                                          validationX, validationy, modelType=self.modelType,
                                          performanceModus="all performances", # "accuracy",
                                          doHyperParameterisation=self.doHyperParameterisation,
                                          hyperParameterRange=self.hyperParameterRange,
                                          hyperParameters=self.hyperParameters,
                                          parametersToAddOrOverwrite=self.parametersToAddOrOverwrite,
                                          nestedModelProp=self.nestedModelProp,
                                          nrOfClasses=self.nrOfClasses)
                modelList.append(model)
        model = ModelEnsambleUtiliser(modelList)
        return model

    def balanceData(self, usedX_train, usedY_train):
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

    def testModelOn(self, X, y):
        pass

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
