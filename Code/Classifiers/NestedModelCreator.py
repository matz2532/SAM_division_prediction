import copy
import numpy as np
import sys
import time
from ModelCreator import ModelCreator
from MyScorer import MyScorer

class NestedModelCreator (ModelCreator):

    def __init__(self, X_train=False, y_train=False, X_test=False, y_test=False,
                modelType="svm", kernel="linear", gamma="scale", C=1.0,
                hyperParameters=None, performanceModus="accuracy",
                doHyperParameterisation=False, hyperParameterRange=None, parametersToAddOrOverwrite=None,
                seed=42, nestedModelProp=False, nrOfClasses=2):
        self.modelType = modelType
        self.setDefaultModelParameters(kernel, gamma, C)
        self.performanceModus = performanceModus
        self.doHyperParameterisation = doHyperParameterisation
        self.nrOfClasses = nrOfClasses
        if self.doHyperParameterisation:
            self.hyperParameters = []
            self.allHyperparameterResults = []
        else:
            self.hyperParameters = hyperParameters
            self.allHyperparameterResults = None
        self.hyperParameterRange = hyperParameterRange
        self.parametersToAddOrOverwrite = parametersToAddOrOverwrite
        self.seed = seed
        np.random.seed(self.seed)
        if isinstance(self.modelType, dict):
            self.createModelTypeParameters()
        self.nestedModelProp = nestedModelProp
        self.initData(X_train, y_train, X_test, y_test)

    def initData(self, X_train, y_train, X_test, y_test):
        if X_train is False or y_train is False:
            self.model = False
            self.trainTime = False
        else:
            self.model, self.trainTime = self.TrainModel(X_train, y_train)
        if self.model is False or X_test is False or y_test is False:
            self.performance = False
        else:
            self.performance = self.TestModel(X_test, y_test)

    def TrainModel(self, X_train, y_train, trainAlreadyGivenModel=False):
        if self.nestedModelProp is False:
            model, trainTime = super().TrainModel(X_train, y_train, trainAlreadyGivenModel=trainAlreadyGivenModel)
        elif self.nestedModelProp is True:
            print("nestedModelProp being True is not implemented in NestedModelCreator yet")
            sys.exit()
        elif type(self.nestedModelProp) == list:
            model, trainTime = self.trainWithSpecificNestedProp(X_train, y_train, trainAlreadyGivenModel=trainAlreadyGivenModel)
        else:
            print("nestedModelProp needs to be False, True, or a nested list.")
        return model, trainTime

    def trainWithSpecificNestedProp(self, X_train, y_train, trainAlreadyGivenModel=False):
        startingTime = time.time()
        nestedModelProp = self.nestedModelProp
        models = []
        currentModel = 0
        allTrainTime = 0
        usedX_train = copy.deepcopy(X_train)
        usedY_train = copy.deepcopy(y_train)
        groupLabel1, groupLabel2 = nestedModelProp[0], nestedModelProp[1]

        isGroup2 = np.isin(usedY_train, groupLabel2)
        usedY_train = np.full(len(usedY_train), 0)
        usedY_train[isGroup2] = 1
        permutedIdx = np.random.permutation(len(usedY_train))
        usedY_train = usedY_train[permutedIdx]
        usedX_train = usedX_train[permutedIdx]
        usedX_train, usedY_train = self.balanceData(usedX_train, usedY_train)
        model, trainTime = self.trainModel(usedX_train, usedY_train,
                                           trainAlreadyGivenModel=trainAlreadyGivenModel,
                                           currentModel=currentModel)
        models.append(model)
        allTrainTime += trainTime
        nestedModelProp = nestedModelProp[1]
        isDataUsed = np.isin(y_train, nestedModelProp)
        usedX_train = copy.deepcopy(X_train[isDataUsed])
        usedY_train = copy.deepcopy(y_train[isDataUsed])
        groupLabel1, groupLabel2 = nestedModelProp[0], nestedModelProp[1]

        isGroup2 = np.isin(usedY_train, groupLabel2)
        usedY_train = np.full(len(usedY_train), 0)
        usedY_train[isGroup2] = 1
        usedX_train, usedY_train = self.balanceData(usedX_train, usedY_train)
        model, trainTime = self.trainModel(usedX_train, usedY_train,
                                           trainAlreadyGivenModel=trainAlreadyGivenModel,
                                           currentModel=currentModel)
        models.append(model)
        allTrainTime += trainTime
        return models, allTrainTime

    def balanceData(self, usedX_train, usedY_train):
        label, counts = np.unique(usedY_train, return_counts=True)
        assert len(counts) == 2, "The number of labels needs to be two. The labels {} are present. {} != 2".format(label, len(counts))
        if counts[0] == counts[1]:
            return usedX_train, usedY_train
        argmaxCount = np.argmax(counts)
        whereIsMajorityLabel = np.where(np.isin(usedY_train, label[argmaxCount]))[0]
        majorityLabelSamplesToRemove = whereIsMajorityLabel[np.min(counts):]
        usedY_train = np.delete(usedY_train, majorityLabelSamplesToRemove)
        usedX_train = np.delete(usedX_train, majorityLabelSamplesToRemove, axis=0)
        label, counts = np.unique(usedY_train, return_counts=True)
        return usedX_train, usedY_train

    def trainModel(self, X_train, y_train, trainAlreadyGivenModel=False, currentModel=0):
        startingTime = time.time()
        if trainAlreadyGivenModel:
            self.model[currentModel].fit(X_train, y_train)
            model = self.model
        elif self.doHyperParameterisation:
            model, hyperParameters, allHyperparameterResults = self.perfromeHyperParameterisation(X_train, y_train, self.hyperParameterRange)
            self.hyperParameters.append(hyperParameters)
            self.allHyperparameterResults.append(allHyperparameterResults)
        elif self.hyperParameters:
            model = self.trainModelWithParameters(self.hyperParameters[currentModel])
        else:
            model = self.setDefaultModel()
            model.fit(X_train, y_train)
            if self.modelType == "k-neighbors":
                self.hyperParameters.append(model.get_params())
                model = model.best_estimator_
        return model, time.time()-startingTime

    def TestModel(self, X_test, y_test, model=None):
        if model is None:
            model = self.model
        if self.nestedModelProp is False:
            predictions = model.predict(X_test)
        elif self.nestedModelProp is True:
            print("nestedModelProp should be a list within a list, with the format of [0, [1, 2]]")
            sys.exit()
        elif type(self.nestedModelProp) == list:
            predictions = self.predictedSpecificNestedProp(X_test, self.nestedModelProp)
        if self.performanceModus == "accuracy":
            accuracy = MyScorer(nrOfClasses=self.nrOfClasses).calcAccuracy(y_test, predictions)
            return accuracy
        else:
            return1DList = self.performanceModus == "all performances 1D list"
            y_scores = model.decision_function(X_test)
            Scorer = MyScorer(y_test, predictions, y_scores=y_scores,
                             nrOfClasses=self.nrOfClasses,
                             setAverage=None if self.nrOfClasses > 2 else "binary")
            performances = Scorer.GetAllScores(return1DList=return1DList)
            return performances

    def predictedSpecificNestedProp(self, X_test, nestedModelProp=None):
        if nestedModelProp is None:
            nestedModelProp = self.nestedModelProp
        y_pred = np.full(len(X_test), 0)
        idxOfYPred = np.arange(len(X_test))

        firstModel = self.model[0]
        currentPredict = firstModel.predict(X_test)
        isFirstLabel = currentPredict == 0
        idxOfFirstLabel = idxOfYPred[isFirstLabel]
        y_pred[idxOfFirstLabel] = nestedModelProp[0]

        # prepare for next round
        isSecondLabel = np.invert(isFirstLabel)
        X_test = X_test[isSecondLabel]
        secondModel = self.model[1]
        currentPredict = secondModel.predict(X_test)
        isFirstLabel = currentPredict == 0
        idxOfYPred = idxOfYPred[isSecondLabel]
        nestedModelProp = nestedModelProp[1]
        y_pred[ idxOfYPred[isFirstLabel] ] = nestedModelProp[0]
        y_pred[ idxOfYPred[np.invert(isFirstLabel)] ] = nestedModelProp[1]
        return y_pred

    def predict(self, X):
        if self.nestedModelProp:
            predictions = self.predictedSpecificNestedProp(X)
        else:
            predictions = self.model.predict(X)
        return predictions

    def GetModel(self):
        return self.model

def balanceTestData(usedX_train, usedY_train):
    label, counts = np.unique(usedY_train, return_counts=True)
    if len(np.unique(counts)) == 1:
        return usedX_train, usedY_train
    sortedIdx = np.argsort(counts)
    for i, sortedIdx in enumerate(sortedIdx):
        if sortedIdx != 0:
            whereIsMajorityLabel = np.where(np.isin(usedY_train, label[i]))[0]
            majorityLabelSamplesToRemove = whereIsMajorityLabel[np.min(counts):]
            usedY_train = np.delete(usedY_train, majorityLabelSamplesToRemove)
            usedX_train = np.delete(usedX_train, majorityLabelSamplesToRemove, axis=0)
            label, counts = np.unique(usedY_train, return_counts=True)
    return usedX_train, usedY_train

def main():
    import pandas as pd
    import sys
    from ModelCreator import ModelCreator
    dataFolder = "Data/WT/topoPredData/ratio/"
    featureFilename = dataFolder + "combinedFeatures_{}_notnormalised.csv".format("topology")
    labelFilename = dataFolder + "combinedLabels.csv"
    features = pd.read_csv(featureFilename, sep=",")
    labels = pd.read_csv(labelFilename, sep=",")
    testPlant = "P2"
    allNestedModelProp = [[0, [1,2]], [1, [0,2]], [2, [0,1]]]
    n = 300
    hyperParameterRange = {"gamma":np.arange(1,10)/10}
    testPlantIdx = np.where(features["plant"] == testPlant)[0]
    features.drop(testPlantIdx,  inplace=True)
    labels.drop(testPlantIdx, inplace=True)
    # for nestedModelProp in allNestedModelProp:
    #     myNestedModelCreator = NestedModelCreator(X_train, y_train, nestedModelProp=nestedModelProp)
    #     myModelCreator = ModelCreator(X_train, y_train, performanceModus="accuracy")
    #     trainAcc = myNestedModelCreator.TestModel(X_train, y_train)
    #     trainFullModelAcc = myModelCreator.TestModel(X_train, y_train)
    #     print(" Train accuracy:", trainAcc, trainFullModelAcc)
    #     valAcc = myNestedModelCreator.TestModel(X_val, y_val)
    #     valFullModelAcc = myModelCreator.TestModel(X_val, y_val)
    #     print("Accuracy:", valAcc, valFullModelAcc)
    nestedModelProp = allNestedModelProp[0]
    print(nestedModelProp)
    X_train, X_val = features.iloc[:-n, 4:].to_numpy(copy=True), features.iloc[-n:, 4:].to_numpy(copy=True)
    y_train, y_val = labels.iloc[:-n, -1].to_numpy(copy=True), labels.iloc[-n:, -1].to_numpy(copy=True)
    X_val, y_val = balanceTestData(X_val, y_val)
    myNestedModelCreator = NestedModelCreator(X_train, y_train, nestedModelProp=nestedModelProp,
                                              doHyperParameterisation=True, hyperParameterRange=hyperParameterRange)
    X_train, y_train = balanceTestData(X_train, y_train)
    myModelCreator = NestedModelCreator(X_train, y_train, performanceModus="accuracy",
                                  doHyperParameterisation=True, hyperParameterRange=hyperParameterRange)
    trainFullModelAcc = myModelCreator.TestModel(X_train, y_train)
    valFullModelAcc = myModelCreator.TestModel(X_val, y_val)
    trainFullNestedModelAcc = myNestedModelCreator.TestModel(X_train, y_train)
    valFullNestedModelAcc = myNestedModelCreator.TestModel(X_val, y_val)
    valPredict = myNestedModelCreator.predictedSpecificNestedProp(X_val)
    testPredict = myNestedModelCreator.predictedSpecificNestedProp(X_train)
    valConfusionMtFullNestedModel = MyScorer().calcTestedConfusionMatrix(y_val, valPredict)
    trainConfusionMtFullNestedModel = MyScorer().calcTestedConfusionMatrix(y_val, valPredict)
    confMtOfModel0 = np.full((2,2),0)
    confMtOfModel0[0,0] = valConfusionMtFullNestedModel[0,0]
    confMtOfModel0[0,1] = np.sum(valConfusionMtFullNestedModel[0,1:3])
    confMtOfModel0[1,0] = np.sum(valConfusionMtFullNestedModel[1:3,0])
    confMtOfModel0[1,1] = np.sum(valConfusionMtFullNestedModel[1:3,1:3])
    valAccModel0 = np.sum(confMtOfModel0*np.eye(2))/np.sum(confMtOfModel0)
    confMtOfModel1 = valConfusionMtFullNestedModel[1:3,1:3]
    valAccModel1 = np.sum(confMtOfModel1*np.eye(2))/np.sum(confMtOfModel1)
    confMtOfModel0 = np.full((2,2),0)
    confMtOfModel0[0,0] = trainConfusionMtFullNestedModel[0,0]
    confMtOfModel0[0,1] = np.sum(trainConfusionMtFullNestedModel[0,1:3])
    confMtOfModel0[1,0] = np.sum(trainConfusionMtFullNestedModel[1:3,0])
    confMtOfModel0[1,1] = np.sum(trainConfusionMtFullNestedModel[1:3,1:3])
    trainAccModel0 = np.sum(confMtOfModel0*np.eye(2))/np.sum(confMtOfModel0)
    confMtOfModel1 = trainConfusionMtFullNestedModel[1:3,1:3]
    trainAccModel1 = np.sum(confMtOfModel1*np.eye(2))/np.sum(confMtOfModel1)
    print("Train accuracy:", trainAccModel0, trainAccModel1, trainFullNestedModelAcc, "simple model train acc:", trainFullModelAcc)
    print("Accuracy:", valAccModel0, valAccModel1, valFullNestedModelAcc, "simple model val acc:", valFullModelAcc)

    print(valConfusionMtFullNestedModel)
    print(np.sum(valConfusionMtFullNestedModel))

if __name__ == '__main__':
    main()
