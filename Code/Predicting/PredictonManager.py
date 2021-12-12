import copy
import numpy as np
import pandas as pd
import pickle
import sys
import time
sys.path.insert(0, "./Code/Analysis Tools/")
sys.path.insert(0, "./Code/Classifiers/")
sys.path.insert(0, "./Code/")

from DataBalancer import DataBalancer
from DividingCellInTableIdentifier import DividingCellInTableIdentifier
from NestedModelCreator import NestedModelCreator
from pathlib import Path
from TopologyPredictonDataCreator import TopologyPredictonDataCreator
from PredictonEvaluator import PredictonEvaluator
from PredictonModelCreator import PredictonModelCreator

class PredictonManager (object):

    verbosity=1

    def __init__(self, plantNames, featureProperty, dataFolder="", sep=",",
                 featureAndLabelFolder="", resultsFolder="Results/Tissue mapping/",
                 testPlants=["P2"], timePointsPerPlant=5, centralCellsDict=None,
                 modelType = {"modelType":"svm","kernel":"rbf"},
                 rebuildData=False, saveRecreatedData=True,
                 nSplits=5,
                 runModelTraining=True, runModelTesting=False,
                 useSpecificTestModelFilename=None,
                 normaliseOnTestData=False,
                 usePreviouslyTrainedModels=False,
                 usePreviousTrainedModelsIfPossible=False,
                 onlyTestModelWithoutTrainingData=False,
                 saveLearningCurve=False,
                 seed=42, specialGraphProperties=None,
                 simplifyLabels=False,
                 useRatio=False, useDifferenceInFeatures=False,
                 useAbsDifferenceInFeatures=False,
                 concatParentFeatures=False, givenFeatureName=None,
                 doHyperParameterisation=False, parametersToAddOrOverwrite=None,
                 hyperParameters=None, specialParName=None,
                 normalisePerTissue=False, normaliseTrainTestData=False,
                 normaliseTrainValTestData=False,
                 maxNormEdgeWeightPerGraph=False,
                 excludeDividingNeighbours=True,
                 nestedModelProp=False, modelEnsambleNumber=1, balanceData=False,
                 modelNameExtension="", folderToSaveVal=None, setFeatureAndLabelFolder=None,
                 useOnlyTwo=False, labelName="combinedLabelsChecked.csv",
                 printPropertyName=False, printModelResults=False,
                 printFeatureAndLabelFilenames=False, printBalancedLabelCount=False,
                 isSaveNormalizedFeatures=True):
        self.allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist"]
        # genral properties
        self.featureProperty = featureProperty
        self.plantNames = plantNames
        self.testPlants = testPlants
        self.timePointsPerPlant = timePointsPerPlant
        self.rebuildData = rebuildData
        self.saveRecreatedData = saveRecreatedData
        self.nSplits = nSplits
        self.simplifyLabels = simplifyLabels
        self.modelType = modelType
        self.dataFolder = dataFolder
        self.featureAndLabelFolder = featureAndLabelFolder
        self.resultsFolder = resultsFolder
        self.sep = sep
        self.useRatio = useRatio
        self.useDifferenceInFeatures = useDifferenceInFeatures
        self.useAbsDifferenceInFeatures = useAbsDifferenceInFeatures
        self.concatParentFeatures = concatParentFeatures
        self.estimatedFeatures=True
        self.estimatedLabels=True
        self.saveLearningCurve = saveLearningCurve
        self.saveDistributionsOfFeatures = False
        self.savePCA = False
        self.printPropertyName = printPropertyName
        self.printFeatureAndLabelFilenames = printFeatureAndLabelFilenames
        self.printModelResults = printModelResults
        self.printBalancedLabelCount = printBalancedLabelCount
        self.isSaveNormalizedFeatures = isSaveNormalizedFeatures
        # self.labelFilenameProperties =
        self.excludeDividingNeighbours = excludeDividingNeighbours
        self.runModelTraining = runModelTraining
        self.runModelTesting = runModelTesting
        self.useSpecificTestModelFilename = useSpecificTestModelFilename
        self.normaliseOnTestData = normaliseOnTestData
        self.usePreviouslyTrainedModels = usePreviouslyTrainedModels
        self.usePreviousTrainedModelsIfPossible = usePreviousTrainedModelsIfPossible
        self.onlyTestModelWithoutTrainingData = onlyTestModelWithoutTrainingData
        self.seed = seed
        # model properties
        self.doHyperParameterisation = doHyperParameterisation
        self.hyperParameterRange = None
        self.parametersToAddOrOverwrite = parametersToAddOrOverwrite
        self.hyperParameters = hyperParameters
        self.specialParName = specialParName
        self.nestedModelProp = nestedModelProp
        self.modelEnsambleNumber = modelEnsambleNumber
        self.balanceData = balanceData
        self.modelNameExtension = modelNameExtension
        self.folderToSaveVal = folderToSaveVal
        self.useOnlyTwo = useOnlyTwo

        # properties for normalisation of features
        self.normalisePerTissue = normalisePerTissue
        self.normaliseTrainTestData = normaliseTrainTestData
        self.normaliseTrainValTestData = normaliseTrainValTestData
        self.normaliseAllData = False

        # underlying network creation properties
        self.maxNormEdgeWeightPerGraph = maxNormEdgeWeightPerGraph
        self.useEdgeWeightInGraph = featureProperty != "topology"
        self.useSharedWallWeightInGraph = featureProperty == "topologyWall"
        self.useDistanceWeight = featureProperty == "topologyDist"
        self.featureNameTemplate = "combinedFeatures_{}_notnormalised.csv"
        self.givenFeatureName = givenFeatureName
        self.labelName = labelName

        if specialGraphProperties is None:
            self.specialGraphProperties = self.createSpecialGraphPorperties()
        else:
            self.specialGraphProperties = specialGraphProperties
            self.maxNormEdgeWeightPerGraph = False
        self.centralCellsDict = centralCellsDict
        self.propertyName = self.definePropertyName()
        if self.printPropertyName:
            print(self.propertyName)
        self.resultsFolder += self.propertyName + "/"
        if self.nestedModelProp:
            self.resultsFolder += str(self.nestedModelProp) + "/"
        Path(self.resultsFolder).mkdir(parents=True, exist_ok=True)
        if self.useRatio:
            self.featureAndLabelFolder += "ratio/"
        elif self.useDifferenceInFeatures:
            self.featureAndLabelFolder += "diff/"
        elif self.useAbsDifferenceInFeatures:
            self.featureAndLabelFolder += "absDiff/"
        else:
            self.featureAndLabelFolder += "default/"
        if self.concatParentFeatures:
            self.featureAndLabelFolder = self.featureAndLabelFolder[:-1]
            self.featureAndLabelFolder += "Concat/"
        if self.centralCellsDict is None:
            self.featureAndLabelFolder += "autoCentralCells/"
        if self.normalisePerTissue:
            self.featureAndLabelFolder += "normalisedPerTissue/"
        if not setFeatureAndLabelFolder is None:
            self.featureAndLabelFolder = setFeatureAndLabelFolder
        Path(self.featureAndLabelFolder).mkdir(parents=True, exist_ok=True)
        if self.givenFeatureName is None:
            print("not given")
            self.featureFilename = self.featureAndLabelFolder + self.featureNameTemplate.format(self.featureProperty)
        else:
            self.featureFilename = self.featureAndLabelFolder + self.givenFeatureName
        self.labelFilename = self.featureAndLabelFolder + self.labelName
        if self.printFeatureAndLabelFilenames:
            print("Feature filename: " + self.featureFilename)
            print("Label filename: " + self.labelFilename)
        self.modelFilename = self.resultsFolder + "models.pkl"
        startTime = time.time()
        self.runManager()
        runTimeInSeconds = time.time() - startTime
        if self.verbosity > 1:
            runTime = runTimeInSeconds / 360
            print("Prediction took {} {}".format(np.round(runTime, 3), "hours"))

    def createSpecialGraphPorperties(self):
        specialGraphPorperties = {}
        specialGraphPorperties["useEdgeWeight"] = self.useEdgeWeightInGraph
        specialGraphPorperties["maxNormEdgeWeightPerGraph"] = self.maxNormEdgeWeightPerGraph
        specialGraphPorperties["useSharedWallWeight"] = self.useSharedWallWeightInGraph
        specialGraphPorperties["useDistanceWeight"] = self.useDistanceWeight
        specialGraphPorperties["invertEdgeWeight"] = self.featureProperty == "topologyArea"
        return specialGraphPorperties

    def definePropertyName(self):
        propertyName = ""
        modelTypeProp = self.defineModelTypeProperty()
        featureProperty = self.defineFeatureTypeProperty()
        propertyName = modelTypeProp + "_" + featureProperty
        return propertyName

    def defineModelTypeProperty(self):
        if "modelType" in self.modelType:
            modelTypeProp = self.modelType["modelType"]
            if "kernel" in self.modelType:
                kernel = self.modelType["kernel"]
                if kernel == "linear":
                    kernelProp = "k1"
                elif kernel == "rbf":
                    kernelProp = "k2"
                elif kernel == "poly":
                    kernelProp = "k3"
                elif kernel == "sigmoid":
                    kernelProp = "k4"
                else:
                    kernelProp = "k0"
            else:
                kernelProp = ""
            if self.doHyperParameterisation:
                hyperParProp = "h"
                if isinstance(self.parametersToAddOrOverwrite, dict):
                    hyperParProp += str(self.parametersToAddOrOverwrite)
            elif not self.specialParName is None:
                hyperParProp = "_{}{}".format(self.specialParName, self.hyperParameters[self.specialParName])
            else:
                hyperParProp = ""
            if hyperParProp or kernelProp:
                modelTypeProp += "_" + kernelProp + hyperParProp
            if self.modelNameExtension:
                modelTypeProp += "{}".format(self.modelNameExtension)
        else:
            modelTypeProp = self.modelType
        return modelTypeProp

    def defineFeatureTypeProperty(self):
        if self.simplifyLabels:
            labelProp = "l2"
        else:
            labelProp = "l3"
        featureUsage = 0
        if self.concatParentFeatures:
            featureUsage += 1
        if self.useRatio:
            featureUsage += 2
        elif self.useDifferenceInFeatures:
            featureUsage += 4
        elif self.useAbsDifferenceInFeatures:
            featureUsage += 4
        featureUsage = "f" + str(featureUsage)
        normalisationCoeff = 0
        if self.normaliseTrainTestData:
            normalisationCoeff += 1
        elif self.normaliseTrainValTestData:
            normalisationCoeff += 4
        if self.normalisePerTissue:
            normalisationCoeff += 2
        if self.maxNormEdgeWeightPerGraph:
            normalisationCoeff += 8
        normalisationCoeff = "n" + str(normalisationCoeff)
        if self.centralCellsDict is None:
            isCetralCellAutomaticallyProp = "c1"
        else:
            isCetralCellAutomaticallyProp = "c0"
        if self.excludeDividingNeighbours:
            excludeDivNeighboursProp = "ex1"
        else:
            excludeDivNeighboursProp = "ex0"
        featureProperty = "{}_{}{}{}{}{}".format(self.featureProperty, labelProp,
                                      featureUsage,  normalisationCoeff,
                                      isCetralCellAutomaticallyProp,
                                      excludeDivNeighboursProp)
        return featureProperty

    def runManager(self):
        existFeatureAndLabels = self.doFeaturesAndLabelsExist()
        if self.featureProperty == "combinedTable" and (not existFeatureAndLabels or self.rebuildData): # not existFeatureAndLabels should actually checking features exit and labels exist somewehere else
            self.createCombinedTable(self.allFeatureProperties)
            self.rebuildData = False
        if self.rebuildData or not existFeatureAndLabels:
            self.dataCreator = TopologyPredictonDataCreator(self.dataFolder,
                                                    self.timePointsPerPlant,
                                                    self.plantNames,
                                                    specialGraphProperties=self.specialGraphProperties,
                                                    centralCellsDict=self.centralCellsDict,
                                                    useRatio=self.useRatio,
                                                    useDifferenceInFeatures=self.useDifferenceInFeatures,
                                                    useAbsDifferenceInFeatures=self.useAbsDifferenceInFeatures,
                                                    concatParentFeatures=self.concatParentFeatures,
                                                    zNormaliseFeaturesPerTissue=self.normalisePerTissue)
            self.dataCreator.MakeTrainingData(estimatedFeatures=self.estimatedFeatures, estimatedLabels=self.estimatedLabels)
            if self.saveRecreatedData:
                self.dataCreator.SaveFeatureTable(self.featureFilename)
                self.dataCreator.SaveLabelTable(self.labelFilename)
            self.features = self.dataCreator.GetFeatureTable()
            self.labels = self.dataCreator.GetLabelTable()
            if self.featureProperty == "combinedTable":
                self.features = pd.read_csv(self.featureFilename, sep=self.sep)
        else:
            self.features = pd.read_csv(self.featureFilename, sep=self.sep)
            self.labels = pd.read_csv(self.labelFilename, sep=self.sep)
        self.features = self.features.copy()
        print(self.features.shape[1])
        self.labels = self.labels.copy()
        self.preProcessFeatruesAndLabels()
        self.hyperParameterRange = self.setHyperParameterRange()
        if not self.hyperParameterRange is None:
            from functools import reduce
            combinations = reduce(lambda x, y: x*len(y), list(self.hyperParameterRange.values()), 1)
            timeInMinutes = combinations / 60
            print("number of tested hyperparameter combinations: {} and estimated time {} in minutes".format(combinations, timeInMinutes))
            # print([len(x) for x in list(self.hyperParameterRange.values())])
        modelCreator = PredictonModelCreator(self.features, self.labels,
                                                     self.testPlants, modelType=self.modelType,
                                                     normaliseTrainTestData=self.normaliseTrainTestData,
                                                     normaliseTrainValTestData=self.normaliseTrainValTestData,
                                                     excludeDividingNeighboursDict=self.excludeDividingNeighboursDict,
                                                     seed=self.seed,
                                                     nSplits=self.nSplits,
                                                     doHyperParameterisation=self.doHyperParameterisation,
                                                     hyperParameterRange=self.hyperParameterRange,
                                                     hyperParameters=self.hyperParameters,
                                                     parametersToAddOrOverwrite=self.parametersToAddOrOverwrite,
                                                     nestedModelProp=self.nestedModelProp,
                                                     modelEnsambleNumber=self.modelEnsambleNumber,
                                                     balanceData=self.balanceData,
                                                     folderToSaveVal=self.resultsFolder,
                                                     useOnlyTwo=self.useOnlyTwo)
        if self.usePreviouslyTrainedModels and self.doHyperParameterisation:
            modelCreator.TrainAndTestModel(runModelTraining=self.runModelTraining,
                            usePreviouslyTrainedModels=pickle.load(open(self.modelFilename, "rb")),
                            printSampleOverview=self.printBalancedLabelCount)
        else:
            modelCreator.TrainAndTestModel(runModelTraining=self.runModelTraining,
                                           usePreviousTrainedModelsIfPossible=self.usePreviousTrainedModelsIfPossible,
                                           normaliseOnTestData=self.normaliseOnTestData,
                                           printSampleOverview=self.printBalancedLabelCount)
        if self.runModelTraining:
            models = modelCreator.GetCVModels()
            trainP = modelCreator.GetTrainPerformance()
            validationP = modelCreator.GetValidationPerformance()
            self.performanceDf = self.createModelResultsTxt(trainP, validationP,
                                    save=True, printOut=self.printModelResults)
            pickle.dump(models, open(self.modelFilename, "wb"))
            self.saveModelProperties()
            if self.doHyperParameterisation:
                self.saveHyperParAndBestEstimator(models)
        else:
            models = self.loadModel()
        X_train, y_train, X_test, y_test = modelCreator.GetTrainTestDataAndLabels()
        if self.isSaveNormalizedFeatures:
            self.saveFeatures(X_train, name="normalizedFeatures_train.csv")
            self.saveFeatures(X_test, name="normalizedFeatures_test.csv")
            self.saveFeatures(y_train, name="labels_train.csv", columnNames=["label"])
            self.saveFeatures(y_test, name="labels_test.csv", columnNames=["label"])
            isCellTestDf = pd.DataFrame({"isCellTest":modelCreator.GetIsCellTest()})
            if self.useOnlyTwo:
                columns = ["plant", "time point", "label"]
            else:
                columns = ["plant", "time point", "dividing parent cell", "parent neighbor", "label"]
            labelOverviewDf = self.labels.loc[:, columns].copy()
            labelOverviewDf.index = np.arange(len(labelOverviewDf))
            labelOverviewDf = pd.concat([labelOverviewDf, isCellTestDf], axis=1)
            labelOverviewDf.to_csv(self.resultsFolder + "labelOverviewDf.csv", index=False)
        if self.runModelTesting:
            if self.onlyTestModelWithoutTrainingData or len(X_train) == 0:
                self.onlyTestModelOnTestData(X_test, y_test, save=True, printOutNrOfTrainTestSamples=self.printBalancedLabelCount)
            else:
                self.testModels(X_train,  y_train, X_test, y_test,
                                printOutNrOfTrainTestSamples=self.printBalancedLabelCount)
        if self.saveLearningCurve:
            if self.doHyperParameterisation:
                hyperparamModels = models
            else:
                hyperparamModels = None
            evaluator = PredictonEvaluator(X_train, y_train,
                                           modelType=self.modelType,
                                           saveToFolder=self.resultsFolder,
                                           seed=self.seed,
                                           hyperparamModels=hyperparamModels,
                                           nestedModelProp=self.nestedModelProp,
                                           modelEnsambleNumber=self.modelEnsambleNumber,
                                           balanceData=self.balanceData,
                                           nrOfClasses= 2 if self.useOnlyTwo is True else 3)
            evaluator.EvaluateModel()
        if self.savePCA:
            baseFolder = "{}PCA/{}/".format(self.dataFolder, self.defineFeatureTypeProperty())
            Path(baseFolder).mkdir(parents=True, exist_ok=True)
            PCA(self.features.iloc[:, 4:], self.labels.iloc[:, -1], doPlot2DPCA=True, showPlot=False, baseFolder=baseFolder)

    def createCombinedTable(self, featurePropertiesToCombine):
        tables = []
        for i, featureProperty in enumerate(featurePropertiesToCombine):
            tableFilename = self.featureAndLabelFolder+self.featureNameTemplate.format(featureProperty)
            table = pd.read_csv(tableFilename)
            if i == 0:
                tables.append(table)
            else:
                tables.append(table.iloc[:, 4:])
        combinedTable = pd.concat(tables, axis=1)
        combineTableFilename = self.featureAndLabelFolder + self.featureNameTemplate.format("combinedTable")
        combinedTable.to_csv(combineTableFilename, index=False)

    def doFeaturesAndLabelsExist(self, printOutIfNotExisting=True):
        isFeatureFile = Path(self.featureFilename).is_file()
        isLabelFile = Path(self.labelFilename).is_file()
        if printOutIfNotExisting:
            if not isFeatureFile:
                print("feature file {} does not exist".format(self.featureFilename))
            if not isLabelFile:
                print("label file {} does not exist".format(self.labelFilename))
        return isFeatureFile and isLabelFile

    def preProcessFeatruesAndLabels(self):
        self.removeDuplicateFeatures()
        columnsToIgnore = ["plant", "time point", "dividing parent cell", "parent neighbor"]
        colIdxToCheck = np.where(np.isin(self.features.columns, columnsToIgnore, invert=True))[0]
        featuresToCheck = self.features.iloc[:, colIdxToCheck]
        if np.any(np.nan == featuresToCheck): # do the same with nas
            columnsContainingZero = np.unique(np.where(np.nan==featuresToCheck)[1])
            columns = np.asarray(list(featuresToCheck.columns))
            self.features.drop(columns[columnsContainingZero], axis=1)
            print("Dropping {} features containing nan".format(columns[columnsContainingZero]))
        columns = np.asarray(list(self.features.columns))
        self.usedFeatures = columns[np.isin(columns, columnsToIgnore, invert=True)]
        self.usedFeatures = list(self.usedFeatures)
        for i, featureName in enumerate(self.usedFeatures):
            if ".1" in featureName:
                self.usedFeatures[i] = featureName.replace(".1", " "+self.allFeatureProperties[1])
            elif ".2" in featureName:
                self.usedFeatures[i] = featureName.replace(".2", " "+self.allFeatureProperties[2])
            elif ".3" in featureName:
                self.usedFeatures[i] = featureName.replace(".3", " "+self.allFeatureProperties[3])
        self.usedFeatures = np.asarray(self.usedFeatures)
        if self.excludeDividingNeighbours:
            self.excludeDividingNeighboursDict = self.createExcludeDividingNeighboursDict()
            self.removeDividingCells()
            # move removal from model creator to here
        else:
            self.excludeDividingNeighboursDict = False
        if self.simplifyLabels:
            labelIsTwo = self.labels.iloc[:, -1] == 2
            self.labels.iloc[np.where(labelIsTwo)[0], -1] = 1
            self.labels.iloc[np.where(np.invert(labelIsTwo))[0], -1] = 0

    def removeDuplicateFeatures(self):
        duplicateColumn = []
        columnNames = list(self.features.columns)
        for i in range(self.features.shape[1]-1):
            for j in range(i+1, self.features.shape[1]):
                if self.areColumnsTheSame(self.features.iloc[:, i], self.features.iloc[:, j]):
                    duplicateColumn.append(columnNames[j])
        duplicateColumn = np.unique(duplicateColumn)
        self.features.drop(duplicateColumn, axis="columns", inplace=True)

    def createExcludeDividingNeighboursDict(self):
        excludeDividingNeighboursDict = {}
        for plantName in self.plantNames:
            dividingCellsPerTimePoint = {}
            for timePoint in np.arange(self.timePointsPerPlant-1):
                parentDaughterLabellingDict = pd.read_csv("{}{}/parentLabeling{}T{}T{}.csv".format(self.dataFolder, plantName, plantName, timePoint, timePoint+1))
                dividingCellsPerTimePoint[timePoint] = parentDaughterLabellingDict
            excludeDividingNeighboursDict[plantName] = dividingCellsPerTimePoint
        return excludeDividingNeighboursDict

    def areColumnsTheSame(self, column1, column2):
        return np.all(column1 == column2)

    def removeDividingCells(self):
        identifier = DividingCellInTableIdentifier(self.features,
                                                   self.excludeDividingNeighboursDict)
        samplesToRemove = identifier.GetAllDividingCellIdxInTable()
        print("nr of samplesToRemove", len(samplesToRemove))
        self.features.drop(samplesToRemove, inplace=True)
        self.labels.drop(samplesToRemove, inplace=True)

    def setHyperParameterRange(self):
        if self.doHyperParameterisation:
            if self.modelType == "random forest":
                return self.setRandomForestHyperParRange()
            else:
                if "modelType" in self.modelType:
                    if self.modelType["modelType"] == "random forest":
                        return self.setRandomForestHyperParRange()
        return None

    def setRandomForestHyperParRange(self):
        hyperParameterRange = {}

        hyperParameterRange["n_estimators"] = np.concatenate([np.arange(10,26,6),[50, 100, 150, 200]])
        hyperParameterRange["max_depth"] = np.concatenate([np.arange(4,11, 4),[15]])
        hyperParameterRange["min_samples_split"] = np.concatenate([np.arange(3,25,4)])
        hyperParameterRange["min_samples_leaf"] = np.concatenate([np.arange(2,16,4)])
        hyperParameterRange["max_features"] = np.concatenate([np.arange(10,35,3)])
        hyperParameterRange["max_leaf_nodes"] = np.concatenate([np.arange(12,17,3)])
        hyperParameterRange["max_samples"] = [0.4]
        return hyperParameterRange

    def loadModel(self):
        doesmModelsExist = Path(self.modelFilename).is_file()
        if doesmModelsExist:
            models = pickle.load(open(self.modelFilename, "rb"))
        else:
            print("model was not found for learning curve cration: {},\nso hyperparamModels is set to None.".format(self.modelFilename))
            models = None
        return models

    def createModelResultsTxt(self, trainP, valP, save=True, printOut=True):
        if not self.useOnlyTwo:
            trainP = [self.combineToF1(i) for i in trainP]
            valP = [self.combineToF1(i) for i in valP]
        columns = self.getPerformanceColumnNames(excludeTrainingPerformance=False)
        trainP = np.asarray(trainP)
        valP = np.asarray(valP)
        try:
            len(trainP[0])
        except:
            trainP = [trainP]
        try:
            len(valP[0])
        except:
            valP = [valP]
        perfromance = np.concatenate([trainP, valP], axis=1)
        perfromanceMean = np.mean(perfromance, axis=0).reshape(1, perfromance.shape[1])
        perfromanceStd = np.std(perfromance, axis=0).reshape(1, perfromance.shape[1])
        performanceDf = np.concatenate([perfromance, perfromanceMean, perfromanceStd], axis=0)
        rowsToRound4Places = [perfromance.shape[1]//2 - 1, perfromance.shape[1] - 1]
        rowsToRound2Places = np.delete(np.arange(perfromance.shape[1]), rowsToRound4Places)
        performanceDf[:, rowsToRound4Places] = np.round(performanceDf[:, rowsToRound4Places], 4)
        performanceDf[:, rowsToRound2Places] = np.round(performanceDf[:, rowsToRound2Places], 2)
        idx = ["split {}".format(i) for i in range(perfromance.shape[0])]
        idx.append("mean")
        idx.append("std")
        performanceDf = pd.DataFrame(performanceDf, columns=columns, index=idx)
        if printOut:
            print(performanceDf.to_string())
        if save:
            performanceDf.to_csv(self.resultsFolder+"results.csv")
        return performanceDf

    def getPerformanceColumnNames(self, excludeTrainingPerformance=False):
        if not self.useOnlyTwo:
            if excludeTrainingPerformance:
                columns = ["val F1", "val c0 F1", "val c1 F1", "val c2 F1", "val Acc", "val precision", "val Auc"]
            else:
                columns = ["train F1", "train c0 F1", "train c1 F1", "train c2 F1", "train Acc", "train precision", "train Auc",
                           "val F1", "val c0 F1", "val c1 F1", "val c2 F1", "val Acc", "val precision", "val Auc"]
        else:
            if excludeTrainingPerformance:
                columns = ["val F1", "val Acc", "val precision", "val Auc"]
            else:
                columns = ["train F1", "train Acc", "train precision", "train Auc",
                           "val F1", "val Acc", "val precision", "val Auc"]
        return columns

    def saveFeatures(self, featureArray, name="normalizedFeatures_train.csv", folder=None, columnNames=None):
        if folder is None:
            folder = self.resultsFolder
        if columnNames is None:
            if self.useOnlyTwo:
                columnNames = self.usedFeatures[1:]
                """
                This is just a fix for the fact that the column cell is selected as column name.
                Does this mean the cell id is passed during training and testing in cell division event prediction?
                """
            else:
                columnNames = self.usedFeatures
        featureTable = pd.DataFrame(featureArray, columns=columnNames)
        filename = folder + name
        featureTable.to_csv(filename, index=False)

    def onlyTestModelOnTestData(self, X_test, y_test, printOut=True, save=True,
                                printOutNrOfTrainTestSamples=True):
        X_test, y_test = DataBalancer(X_test, y_test).GetBalancedData()
        self.dumpTestXAndYs(X_test, y_test)
        if printOutNrOfTrainTestSamples:
            print("test labels:" , np.unique(y_test, return_counts=True))
        if self.useSpecificTestModelFilename:
            testModelFilename = self.useSpecificTestModelFilename
        else:
            testModelFilename = self.resultsFolder + "testModel.pkl"
        print(testModelFilename)
        assert Path(testModelFilename).is_file(), "The filename {} of the model does not exist.".format(testModelFilename)
        myModelCreator = pickle.load(open(testModelFilename, "rb"))
        testP = myModelCreator.TestModel(X_test, y_test)
        columns = self.getPerformanceColumnNames(excludeTrainingPerformance=True)
        performance = np.asarray(testP)
        performanceLen = len(performance)
        rowsToRound4Places = [performanceLen//2 - 1, performanceLen - 1]
        rowsToRound2Places = np.delete(np.arange(performanceLen), rowsToRound4Places)
        performance[rowsToRound4Places] = np.round(performance[rowsToRound4Places], 4)
        performance[rowsToRound2Places] = np.round(performance[rowsToRound2Places], 2)
        idx = ["testing"]
        performanceDf = pd.DataFrame([performance], columns=columns, index=idx)
        if printOut:
            print(performanceDf.to_string())
        if save:
            performanceDf.to_csv(self.resultsFolder+"resultsWithTesting.csv")
        return performanceDf

    def dumpTestXAndYs(self, X_test, y_test):
        pickle.dump([X_test], open(self.resultsFolder+"testXs.pkl", "wb"))
        pickle.dump([y_test], open(self.resultsFolder+"testYs.pkl", "wb"))

    def testModels(self, X_train, y_train, X_test, y_test, redoModel=True,
                   perfromanceTrainValDf=None, printOut=False, save=True,
                   printOutNrOfTrainTestSamples=True):
        if self.balanceData:
            X_train, y_train = DataBalancer(X_train, y_train).GetBalancedData()
            X_test, y_test = DataBalancer(X_test, y_test).GetBalancedData()
        self.dumpTestXAndYs(X_test, y_test)
        if printOutNrOfTrainTestSamples:
            print("train labels:" , np.unique(y_train, return_counts=True))
            print("test labels:" , np.unique(y_test, return_counts=True))
        if self.useSpecificTestModelFilename:
            testModelFilename = self.useSpecificTestModelFilename
        else:
            testModelFilename = self.resultsFolder + "testModel.pkl"
        if Path(testModelFilename).is_file() and not redoModel:
            myModelCreator = pickle.load(open(testModelFilename, "rb"))
        else:
            myModelCreator = NestedModelCreator(X_train, y_train,
                                                performanceModus="all performances 1D list",
                                                doHyperParameterisation=True,
                                                nrOfClasses=2 if self.useOnlyTwo is True else 3)
            pickle.dump(myModelCreator, open(testModelFilename, "wb"))
        trainP = myModelCreator.TestModel(X_train, y_train)
        testP = myModelCreator.TestModel(X_test, y_test)
        columns = self.getPerformanceColumnNames(excludeTrainingPerformance=True)
        if perfromanceTrainValDf is None:
            perfromanceTrainValDf = pd.read_csv(self.resultsFolder+"results.csv", index_col=0)
        trainP = np.asarray(trainP)
        testP = np.asarray(testP)
        performance = np.concatenate([trainP, testP])
        performanceLen = len(performance)
        rowsToRound4Places = [performanceLen//2 - 1, performanceLen - 1]
        rowsToRound2Places = np.delete(np.arange(performanceLen), rowsToRound4Places)
        performance[rowsToRound4Places] = np.round(performance[rowsToRound4Places], 4)
        performance[rowsToRound2Places] = np.round(performance[rowsToRound2Places], 2)
        idx = ["testing"]
        performanceDf = pd.DataFrame([performance], columns=columns, index=idx)
        performanceDf = pd.concat([perfromanceTrainValDf, performanceDf], axis=0)
        if printOut:
            print(performanceDf.to_string())
        if save:
            performanceDf.to_csv(self.resultsFolder+"resultsWithTesting.csv")
        return performanceDf

    def combineToF1(self, performance):
        multiF1 = performance[0]
        p = [np.mean(multiF1)]
        p.extend([i for i in multiF1])
        p.extend(performance[1:])
        return p

    def saveModelProperties(self):
        file = open(self.resultsFolder+"modelProperty.txt", "w")
        file.write("propertyName: "+self.propertyName+",\n")
        file.write("featureFilename: "+self.featureFilename+",\n")
        file.write("labelFilename: "+self.labelFilename+",\n")
        labels, counts = np.unique(self.labels.iloc[:,-1], return_counts=True)
        nrOfLabels = {key:value for key, value in zip(labels, counts)}
        file.write("nrOfLabels: "+str(nrOfLabels))
        file.write("nrOfFeatures: "+str(self.features.shape[1]))
        file.close()

    def saveHyperParAndBestEstimator(self, models, hyperParName="hyperParRange.txt"):
        bestEstimators, param_grid = self.getBesEstimatorAndParamGrid(models)
        hyperParTxt = "{"
        for parName, parVal in param_grid.items():
            hyperParTxt += "{}: {}\n".format(parName, parVal)
        hyperParTxt = hyperParTxt[:-1]
        hyperParTxt += "}\n"
        for estimatorParameter in bestEstimators:
            hyperParTxt += "{}\n".format(estimatorParameter)
        file = open(self.resultsFolder + hyperParName, "w")
        file.write(hyperParTxt)
        file.close()

    def getBesEstimatorAndParamGrid(self, splitsOfEnsambles):
        modelId = 0
        bestEstimators = []
        for splitNr in range(len(splitsOfEnsambles)):
            ensamble = splitsOfEnsambles[splitNr]
            if type(ensamble) == list:
                model = ensamble[splitNr].GetModel()
            else:
                model = ensamble
            param_grid = model.param_grid
            bestEstimators.append(model.best_estimator_)
        return bestEstimators, param_grid

    def GetPerfromanceDf(self):
        return self.performanceDf

def main():
    dataFolder = "Data/WT/"
    plantNames = ["P1", "P2", "P5", "P6", "P8"]
    testPlants = ["P2"]
    modelType =  {"modelType":"svm","kernel":"rbf"}
    useTemporaryResultsFolder = False
    runDivEventPred = False
    runModelTraining = True
    runModelTesting = True
    onlyTestModelWithoutTrainingData = False
    saveLearningCurve = True
    useManualCentres = True
    # print options:
    printBalancedLabelCount = True
    nestedModelProp = False
    balanceData = False
    hyperParameters = None
    modelNameExtension = ""
    centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    doHyperParameterisation = True
    normalisePerTissue = False
    normaliseTrainTestData = True
    normaliseTrainValTestData = False
    featureProperty = "combinedTable"
    if runDivEventPred:
        for set in ["allTopos", "area", "topoAndBio", "lowCor0.5", "lowCor0.7", "topology"]:
            labelName = "combinedLabels.csv"
            if useManualCentres:
                setFeatureAndLabelFolder = "Data/WT/divEventData/manualCentres/{}/".format(set)
                resultsFolder = "Results/divEventData/manualCentres/{}/".format(set)
            else:
                setFeatureAndLabelFolder = "Data/WT/divEventData/{}/".format(set)
                resultsFolder = "Results/divEventData/{}/".format(set)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("newResultsFolder: "+newResultsFolder)
            manager = PredictonManager(plantNames=plantNames,
                                   testPlants=testPlants,
                                   featureProperty=featureProperty,
                                   dataFolder=dataFolder,
                                   featureAndLabelFolder=setFeatureAndLabelFolder,
                                   givenFeatureName=givenFeatureName,
                                   resultsFolder=newResultsFolder,
                                   modelType=modelType,
                                   runModelTraining=runModelTraining,
                                   runModelTesting = runModelTesting,
                                   onlyTestModelWithoutTrainingData=onlyTestModelWithoutTrainingData,
                                   saveLearningCurve=saveLearningCurve,
                                   centralCellsDict=centralCellsDict,
                                   normalisePerTissue=normalisePerTissue,
                                   normaliseTrainTestData=normaliseTrainTestData,
                                   normaliseTrainValTestData=normaliseTrainValTestData,
                                   doHyperParameterisation=doHyperParameterisation,
                                   hyperParameters=hyperParameters,
                                   nestedModelProp=nestedModelProp,
                                   balanceData=balanceData,
                                   modelNameExtension=modelNameExtension,
                                   folderToSaveVal=folderToSaveVal,
                                   setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                   useOnlyTwo=True,
                                   labelName=labelName,
                                   excludeDividingNeighbours=False,
                                   printBalancedLabelCount=printBalancedLabelCount)
        sys.exit()
    for excludeDividingNeighbours in [True, False]:
        for set in ["allTopos", "bio", "topoAndBio", "topology", "lowCor0.7", "lowCor0.5"]:
            labelName = "combinedLabels.csv"
            if useManualCentres:
                setFeatureAndLabelFolder = "Data/WT/topoPredData/diff/manualCentres/{}/".format(set)
                resultsFolder = "Results/topoPredData/diff/manualCentres/{}/".format(set)
            else:
                setFeatureAndLabelFolder = "Data/WT/topoPredData/diff/{}/".format(set)
                resultsFolder = "Results/topoPredData/diff/{}/".format(set)
            if useTemporaryResultsFolder:
                resultsFolder = "Temporary/{}/".format(set)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("newResultsFolder: "+newResultsFolder)
            manager = PredictonManager(plantNames=plantNames,
                                       testPlants=testPlants,
                                       featureProperty=featureProperty,
                                       dataFolder=dataFolder,
                                       featureAndLabelFolder=setFeatureAndLabelFolder,
                                       givenFeatureName=givenFeatureName,
                                       resultsFolder=newResultsFolder,
                                       modelType=modelType,
                                       runModelTraining = runModelTraining,
                                       runModelTesting = runModelTesting,
                                       saveLearningCurve = saveLearningCurve,
                                       centralCellsDict=centralCellsDict,
                                       normalisePerTissue=normalisePerTissue,
                                       normaliseTrainTestData=normaliseTrainTestData,
                                       normaliseTrainValTestData=normaliseTrainValTestData,
                                       doHyperParameterisation=doHyperParameterisation,
                                       hyperParameters=hyperParameters,
                                       nestedModelProp=nestedModelProp,
                                       balanceData=balanceData,
                                       modelNameExtension=modelNameExtension,
                                       folderToSaveVal=folderToSaveVal,
                                       setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                       labelName=labelName,
                                       excludeDividingNeighbours=excludeDividingNeighbours,
                                       printBalancedLabelCount=printBalancedLabelCount)

if __name__ == '__main__':
    main()
