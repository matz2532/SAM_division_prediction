import math
import numpy as np
import sys
import time
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from MyScorer import MyScorer

class ModelCreator (object):

    def __init__(self, X_train=False, y_train=False, X_test=False, y_test=False,
                modelType="svm", kernel="linear", gamma="scale", C=1.0,
                hyperParameters=None, performanceModus=None,
                doHyperParameterisation=False, hyperParameterRange=None, parametersToAddOrOverwrite=None,
                seed=42):
        self.modelType = modelType
        self.setDefaultModelParameters(kernel, gamma, C)
        self.hyperParameters = hyperParameters
        self.allHyperparameterResults = None
        self.performanceModus = performanceModus
        self.doHyperParameterisation = doHyperParameterisation
        self.hyperParameterRange = hyperParameterRange
        self.parametersToAddOrOverwrite = parametersToAddOrOverwrite
        self.seed = seed
        if isinstance(self.modelType, dict):
            self.setModelTypeParameters()
        self.initData(X_train, y_train, X_test, y_test)

    def setDefaultModelParameters(self, kernel, gamma, C):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

    def setModelTypeParameters(self):
        if "kernel" in self.modelType:
            self.kernel = self.modelType["kernel"]
        if "gamma" in self.modelType:
            self.gamma = self.modelType["gamma"]
        if "C" in self.modelType:
            self.C = self.modelType["C"]
        if "modelType" in self.modelType:
            self.modelType = self.modelType["modelType"]

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

    def TrainModel(self, X_train, y_train, trainSetModel=False):
        startingTime = time.time()
        if trainSetModel:
            self.model.fit(X_train, y_train)
            model = self.model
        elif self.doHyperParameterisation:
            model, self.hyperParameters, self.allHyperparameterResults = self.perfromeHyperParameterisation(X_train, y_train, self.hyperParameterRange)
        elif self.hyperParameters:
            model = self.trainModelWithParameters(self.hyperParameters)
            model.fit(X_train, y_train)
        else:
            print("This is not yet implemented.")
            sys.exit()
        return model, time.time()-startingTime

    def perfromeHyperParameterisation(self, X_train, y_train, parameters=None):
        if parameters is None:
            parameters = self.calcDefaultHyperPar(X_train)
        assert self.parametersToAddOrOverwrite is None or isinstance(self.parametersToAddOrOverwrite, dict), "parametersToAddOrOverwrite needs to be either None or a dictionary, {} != dict. It looks like: {}".format(type(parametersToAddOrOverwrite), parametersToAddOrOverwrite)
        if isinstance(self.parametersToAddOrOverwrite, dict):
            parameters = self.addOrOverwriteParameters(parameters)
        model = self.setDefaultModel()
        model = GridSearchCV(model, parameters, scoring="accuracy", cv=5, n_jobs=100, verbose=1)
        model.fit(X_train, y_train)
        return model, model.best_params_, model.cv_results_

    def calcDefaultHyperPar(self, X_train):
        if self.kernel == "rbf":
            density = 101
            gamma = np.concatenate([self.equallySpacedValueSamplingOverScales([-8,-7], density), self.equallySpacedValueSamplingOverScales([-5, -5], density)])
            C = np.concatenate([self.equallySpacedValueSamplingOverScales([1,1], density), self.equallySpacedValueSamplingOverScales([4, 4], density)])
            parameters = {'gamma' : gamma, 'C' : C}
        else:
            print("The kernel {} is not yet implemented.".format(self.kernel))
            sys.exit()
        return parameters

    def addOrOverwriteParameters(self, parameters):
        for parameterName, parametersToAddOrOverwrite in self.parametersToAddOrOverwrite:
            if parameterName in parameters:
                print("Before", parameters[parameterName])
            parameters[parameterName] = parametersToAddOrOverwrite
            print("After", parameters[parameterName])
        return parameters

    def equallySpacedValueSamplingOverScales(self, minMaxScale=[-10, -5], nrOfSamplesPerScale=10):
        parameters = []
        includeLast = False
        nrOfSamplesPerScale -= 1
        for i in range(minMaxScale[0], minMaxScale[1]+1):
            if i == minMaxScale[1]:
                includeLast = True
                nrOfSamplesPerScale += 1
            par = np.linspace(1*math.pow(10, i), 10*math.pow(10, i), nrOfSamplesPerScale, includeLast)
            parameters.append(par)
        parameters = np.concatenate(parameters)
        return parameters

    def setDefaultModel(self):
        if self.modelType == "svm":
            model = svm.SVC(class_weight="balanced", kernel=self.kernel, gamma=self.gamma, C=self.C)
        else:
            print("The model type {} is not yet implemented.".format(self.modelType))
            sys.exit()
        return model

    def trainModelWithParameters(self, hyperParameters):
        if self.modelType == "svm":
            kernel = hyperParameters["kernel"] if "kernel" in hyperParameters else self.kernel
            if "gamma" in hyperParameters:
                model = svm.SVC(kernel=kernel, gamma=hyperParameters["gamma"], C=hyperParameters["C"])
            else:
                model = svm.SVC(kernel=kernel, C=hyperParameters["C"])
        else:
            print("The model type {} is not yet implemented.".format(self.modelType))
            sys.exit()
        return model

    def TestModel(self, X_test, y_test, model=None):
        if model is None:
            model = self.model
        predictions = model.predict(X_test)
        if self.performanceModus == "accuracy":
            accuracy = MyScorer(nrOfClasses=self.nrOfClasses).calcAccuracy(y_test, predictions)
            return accuracy
        else:
            y_scores = model.decision_function(X_test)
            f1Score, accuracy, precision, auc = MyScorer(y_test, predictions,
                                                         y_scores=y_scores,
                                                         nrOfClasses=self.nrOfClasses
                                                         ).GetAllScores()
            return [f1Score, accuracy, precision, auc]

    def GetModel(self):
        if not self.model is False:
            return self.model
        else:
            print("No model was calculated yet.")
            sys.exit(1)

    def GetPerfromance(self):
        if not self.performance is False:
            return self.performance
        else:
            print("No performance was calculated yet. Provide x_train and y_train during initialistion.")
            sys.exit(1)

    def GetModelTrainingTime(self):
        if not self.trainTime is False:
            return self.trainTime
        else:
            print("No model was calculated yet.")
            sys.exit(1)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def decision_function(self, X_test):
        y_scores = self.model.decision_function(X_test)
        return y_scores

    def GetHyperParameters(self):
        if self.hyperParameters:
            return self.hyperParameters
        print("No hyper-parameters are calculated/used and None is returned.")
        return None

    def GetAllHyperParameters(self):
        return self.allHyperparameterResults

    def SetModel(self, newModel):
        self.model = newModel

def main():
    # add own gaussion kernel use code from stack overflow 26962159 "How to use a custom SVM kernel?"
    myModelCreator = ModelCreator()

if __name__ == '__main__':
    main()
