import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys

sys.path.insert(0, "./Code/")
sys.path.insert(0, "./Code/Analysis Tools/")
sys.path.insert(0, "./Code/Classifiers/")

from LearningCurvePlotter import LearningCurvePlotter
from BalancedXFold import BalancedXFold
from NestedModelCreator import NestedModelCreator
from pathlib import Path
from PerPlantFold import PerPlantFold
from utils import doZNormalise

class PredictonEvaluator (object):

    verbosity=2

    def __init__(self, X_train, y_train, modelType, X_test=None, y_test=None,
                 nSplits=5, seed=42, plantNameOfTrainData=None, startRange=8,
                 useTestData=False, saveToFolder=None,
                 hyperparamModels=None, nestedModelProp=False,
                 modelEnsambleNumber=False,
                 balanceData=False, selectedData=1, nrOfClasses=3,
                 printLearningCurveSplit=True):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.seed = seed
        self.nSplits = nSplits
        self.plantNameOfTrainData = plantNameOfTrainData
        self.useTestData = useTestData
        self.modelType = modelType
        self.saveToFolder = saveToFolder
        self.startRange = startRange
        self.hyperparamModels = hyperparamModels
        self.nestedModelProp = nestedModelProp
        self.modelEnsambleNumber = modelEnsambleNumber
        self.selectedData = selectedData
        self.balanceData = balanceData
        self.nrOfClasses = nrOfClasses
        self.printLearningCurveSplit = printLearningCurveSplit
        np.random.seed(self.seed)

    def EvaluateModel(self, useTestData=None):
        if not useTestData is None:
            self.useTestData = useTestData
        self.doLearningCurveEstimation(self.useTestData)
        if useTestData:
            self.testModelOn(self.X_test, self.y_test)

    def doLearningCurveEstimation(self, useTestData=False, loadPerformances=False):
        allTrainPsFilename = self.saveToFolder + "allTrainPsStartingFrom{}.pkl".format(self.startRange)
        allValPsFilename = self.saveToFolder + "allValPsStartingFrom{}.pkl".format(self.startRange)
        usedSmpleNumbersFilename = self.saveToFolder + "usedSmpleNumbersStartingFrom{}.pkl".format(self.startRange)
        allPsStartingFromExist = Path(allTrainPsFilename).is_file() and Path(allValPsFilename).is_file()
        if not loadPerformances or not allPsStartingFromExist:
            allTrainPs, allValPs, usedSmpleNumbers = self.calcTrainAndValPerformances(self.startRange)
            pickle.dump(allTrainPs, open(allTrainPsFilename, "wb"))
            pickle.dump(allValPs, open(allValPsFilename, "wb"))
            pickle.dump(usedSmpleNumbers, open(usedSmpleNumbersFilename, "wb"))
        else:
            allTrainPs = pickle.load(open(allTrainPsFilename, "rb"))
            allValPs = pickle.load(open(allValPsFilename, "rb"))
            usedSmpleNumbers = pickle.load(open(usedSmpleNumbersFilename, "rb"))
        assert allValPs.shape[0] == 7 or allValPs.shape[0] == 4, "Assuming that there are 5 or 8 rows/performance measures, {} != 4 or != 7".format(allValPs.shape[0])
        LearningCurvePlotter(allValPs[1,:,:], allTrainPs[1,:,:], usedSmpleNumbers,
                             yLabel="Accuracy [%]",
                             showPlot=self.saveToFolder is None,
                             testLabel="Validation Score",
                             limitYAxis=False)
        if not self.saveToFolder is None:
            plt.savefig(self.saveToFolder+"learning curve.png", bbox_inches="tight")
        else:
            plt.show()

    def calcTrainAndValPerformances(self, startRange=20, stepSize=5):
        allTrainPs, allValPs = [], []
        allTrainLengths = []
        currentSplit = 0
        doPerPlantSplit = type(self.nSplits) != int
        if doPerPlantSplit:
            assert not self.plantNameOfTrainData is None, "The plantNameOfTrainData needs to be defined, when doing per plant name split."
            xFold = PerPlantFold(plantNameVector=self.plantNameOfTrainData, balanceData=self.balanceData)
            uniquePlantNames = xFold.GetUniquePlantNames()
        else:
            if self.balanceData:
                xFold = BalancedXFold(n_splits=self.nSplits, random_state=self.seed)
            else:
                raise NotImplementedError
        for trainIdx, validationIdx in xFold.split(self.X_train, self.y_train):
            if self.printLearningCurveSplit or self.verbosity >= 1:
                if doPerPlantSplit:
                    print("val plant {} with split {}/{} of learning curve calculation".format(uniquePlantNames[currentSplit], currentSplit+1, len(uniquePlantNames)))
                else:
                    print("Split {}/{} of learning curve calculation".format(currentSplit+1, self.nSplits))
            X_train = self.X_train[trainIdx, :]
            y_train = self.y_train[trainIdx]
            X_val = self.X_train[validationIdx, :]
            y_val = self.y_train[validationIdx]
            X_train, X_val = self.normaliseFeatures(X_train, X_val)
            allTrainLengths.append(len(X_train))
            if self.hyperparamModels is None:
                modelWithParamsSet = None
            else:
                modelWithParamsSet = self.hyperparamModels[currentSplit].best_estimator_
            trainPs, valPs = self.calcScoreForRange(X_train, y_train,
                                                  X_val, y_val,
                                                  minMaxRange=[startRange, len(X_train)],
                                                  stepSize=stepSize,
                                                  modelWithParamsSet=modelWithParamsSet)
            allTrainPs.append(trainPs)
            allValPs.append(valPs)
            currentSplit += 1
        if not self.areTrainLengthEqual(allTrainLengths):
            allTrainPs = self.trimInternalListLengthToFloor(allTrainPs)
            allValPs = self.trimInternalListLengthToFloor(allValPs)
        max = startRange + stepSize*len(allTrainPs[0])
        usedSmpleNumbers = np.arange(startRange, max, stepSize)
        allTrainPs = np.asarray(allTrainPs).T
        allValPs = np.asarray(allValPs).T
        return allTrainPs, allValPs, usedSmpleNumbers

    def normaliseFeatures(self, X_train, X_val):
        X_train, normParameters = doZNormalise(X_train, returnParameter=True)
        X_val = doZNormalise(X_val, useParameters=normParameters)
        return X_train, X_val

    def calcScoreForRange(self, X_train, y_train, X_val, y_val, minMaxRange,
                          stepSize=1, modelWithParamsSet=None):
        allTrainP = []
        allValP = []
        trainSampleRange = np.arange(minMaxRange[0], minMaxRange[1], stepSize)
        for i, trainSampleSize in enumerate(trainSampleRange):
            if self.verbosity == 2:
                if i % 100 == 0:
                    print("trainSampleSize at {}; {}/{}".format(trainSampleSize, i+1, len(trainSampleRange)))
            elif self.verbosity >= 2:
                print("trainSampleSize", trainSampleSize)
            selectedSamples = np.arange(len(X_train))
            numberOfUnqiues = len(np.unique(y_train))
            shuffleRun = 0
            while shuffleRun == 0 or numberOfUnqiues > len(np.unique(y_train[selectedSamples[:trainSampleSize]])):
                shuffleRun += 1
                np.random.shuffle(selectedSamples)
                if shuffleRun > 20:
                    print("shuffled 20 times")
                    break
            assert numberOfUnqiues == len(np.unique(y_train[selectedSamples[:trainSampleSize]])), "The samples are reshuffled but the selected labels don't contain at least {} different labels {}".format(numberOfUnqiues, np.unique(y_train[selectedSamples[:trainSampleSize]]))
            selectedSamples = selectedSamples[:trainSampleSize]
            selectedX_train = X_train[selectedSamples, :]
            selectedy_train = y_train[selectedSamples]
            if modelWithParamsSet is None:
                myModelCreator = NestedModelCreator(selectedX_train,
                                              selectedy_train,
                                              modelType=self.modelType,
                                              performanceModus="accuracy",
                                              nestedModelProp=self.nestedModelProp,
                                              nrOfClasses=self.nrOfClasses)
            else:
                myModelCreator = NestedModelCreator(performanceModus="all performances 1D list",
                                                    nestedModelProp=self.nestedModelProp,
                                                    nrOfClasses=self.nrOfClasses)
                myModelCreator.SetModel(modelWithParamsSet)
                myModelCreator.TrainModel(selectedX_train,
                                          selectedy_train,
                                          trainAlreadyGivenModel=True)
            trainPerformance = myModelCreator.TestModel(selectedX_train,  selectedy_train)
            valPerformance = myModelCreator.TestModel(X_val, y_val)
            allTrainP.append(trainPerformance)
            allValP.append(valPerformance)
        return allTrainP, allValP

    def areTrainLengthEqual(self, allTrainLengths):
        isLengthWithSuccessorEqual = [allTrainLengths[i] == allTrainLengths[i+1] for i in range(len(allTrainLengths)-1)]
        return np.all(isLengthWithSuccessorEqual)

    def trimInternalListLengthToFloor(self, nestedList):
        lengthsOfInternalLists = [len(internalList) for internalList in nestedList]
        floor = np.min(lengthsOfInternalLists)
        for i, internalList in enumerate(nestedList):
            nestedList[i] = internalList[:floor]
        return nestedList

    def testModelOn(self, X_test, y_test):
        print("Testing is not yet implemented and will be done at the last step.")
