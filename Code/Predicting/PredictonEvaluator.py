import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.insert(0, "./Code/Analysis Tools/")
sys.path.insert(0, "./Code/Classifiers/")

from LearningCurvePlotter import LearningCurvePlotter
from BalancedXFold import BalancedXFold
from NestedModelCreator import NestedModelCreator
from pathlib import Path

class PredictonEvaluator (object):

    def __init__(self, X_train, y_train, modelType, X_test=None, y_test=None,
                 nSplits=5, seed=42, startRange=8,
                 useTestData=False, saveToFolder=None,
                 hyperparamModels=None, nestedModelProp=False,
                 modelEnsambleNumber=False,
                 selectedData=1, nrOfClasses=3,
                 printLearningCurveSplit=True):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.seed = seed
        self.nSplits = nSplits
        self.useTestData = useTestData
        self.modelType = modelType
        self.saveToFolder = saveToFolder
        self.startRange = startRange
        self.hyperparamModels = hyperparamModels
        self.nestedModelProp = nestedModelProp
        self.modelEnsambleNumber = modelEnsambleNumber
        self.selectedData = selectedData
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
        allPsStartingFromExist = Path(allTrainPsFilename).is_file() and Path(allValPsFilename).is_file()
        if not loadPerformances or not allPsStartingFromExist:
            allTrainPs, allValPs = self.calcTrainAndValPerformances(self.startRange)
            pickle.dump(allTrainPs, open(allTrainPsFilename, "wb"))
            pickle.dump(allValPs, open(allValPsFilename, "wb"))
        else:
            allTrainPs = pickle.load(open(allTrainPsFilename, "rb"))
            allValPs = pickle.load(open(allValPsFilename, "rb"))
        assert allValPs.shape[0] == 8 or allValPs.shape[0] == 5, "Assuming that there are 5 or 8 rows/performance measures, {} != 5 or != 8".format(allValPs.shape[0])
        # doing this to ensure -4th row is accuracy which is used in the learning curve
        sampleNrRange = np.arange(self.startRange, self.startRange + allTrainPs.shape[1])
        LearningCurvePlotter(allValPs[-4,:,:], allTrainPs[-4,:,:], sampleNrRange,
                             yLabel="Accuracy [%]",
                             showPlot=self.saveToFolder is None,
                             testLabel="Validation Score",
                             limitYAxis=False)
        if not self.saveToFolder is None:
            plt.savefig(self.saveToFolder+"learning curve.png", bbox_inches="tight")
        else:
            plt.show()

    def calcTrainAndValPerformances(self, startRange=10):
        allTrainPs, allValPs = [], []
        allTrainLengths = []
        currentSplit = 0
        xFold = BalancedXFold(n_splits=self.nSplits, random_state=self.seed)
        for trainIdx, validationIdx in xFold.split(self.X_train, self.y_train):
            if self.printLearningCurveSplit:
                print("Split {}/{} of learning curve calculation".format(currentSplit+1, self.nSplits))
            X_train = self.X_train[trainIdx, :]
            y_train = self.y_train[trainIdx]
            X_val = self.X_train[validationIdx, :]
            y_val = self.y_train[validationIdx]
            print(currentSplit,"startRange", startRange, "train:", len(y_train), "val:", len(y_val))
            allTrainLengths.append(len(X_train))
            if self.hyperparamModels is None:
                modelWithParamsSet = None
            else:
                modelWithParamsSet = self.hyperparamModels[currentSplit]
                if self.modelEnsambleNumber > 1 and self.nestedModelProp:
                    modelWithParamsSet = [[model.best_estimator_ for model in nestedModel] for nestedModel in modelWithParamsSet]
                elif self.nestedModelProp or self.modelEnsambleNumber > 1:
                    modelWithParamsSet = [model.best_estimator_ for model in modelWithParamsSet]
                else:
                    modelWithParamsSet = modelWithParamsSet.best_estimator_
            trainPs, valPs = self.calcScoreForRange(X_train, y_train,
                                                  X_val, y_val,
                                                  minMaxRange=[startRange, len(X_train)],
                                                  modelWithParamsSet=modelWithParamsSet)
            allTrainPs.append(trainPs)
            allValPs.append(valPs)
            currentSplit += 1
        if not self.areTrainLengthEqual(allTrainLengths):
            allTrainPs = self.trimInternalListLengthToFloor(allTrainPs)
            allValPs = self.trimInternalListLengthToFloor(allValPs)
        allTrainPs = np.asarray(allTrainPs).T
        allValPs = np.asarray(allValPs).T
        return allTrainPs, allValPs

    def calcScoreForRange(self, X_train, y_train, X_val, y_val, minMaxRange, modelWithParamsSet=None):
        allTrainP = []
        allValP = []
        for trainSampleSize in range(minMaxRange[0], minMaxRange[1]):
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
                # employ different samples for each model training same as PredictonModelCreator
                if self.nestedModelProp:
                    modelList = []
                    for i, model in enumerate(modelWithParamsSet):
                        retrainedModel = NestedModelCreator(performanceModus="accuracy",
                                                            nestedModelProp=self.nestedModelProp,
                                                            nrOfClasses=self.nrOfClasses)
                        myModelCreator.SetModel(modelWithParamsSet)
                        myModelCreator.TrainModel(selectedX_train,
                                                  selectedy_train,
                                                  trainSetModel=True)
                        modelList.append(retrainedModel)
                    myModelCreator = ModelEnsambleUtiliser(modelList)
                else:
                    myModelCreator = NestedModelCreator(performanceModus="all performances 1D list",
                                                        nestedModelProp=self.nestedModelProp,
                                                        nrOfClasses=self.nrOfClasses)
                    myModelCreator.SetModel(modelWithParamsSet)
                    myModelCreator.TrainModel(selectedX_train,
                                              selectedy_train,
                                              trainSetModel=True)
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

def main():
    selectedFeature = 0
    dataFolder = "Data/WT/topoPredData/ratio/"
    allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist"]
    features = dataFolder + "combinedFeatures_{}_notnormalised.csv".format(allFeatureProperties[selectedFeature])
    labels = dataFolder + "combinedLabels.csv"
    features = pd.read_csv(features)
    labels = pd.read_csv(labels)
    doPCA = True
    evaluator = PredictonEvaluator(features, labels)
    evaluator.PlotPCA()

if __name__ == '__main__':
    main()
