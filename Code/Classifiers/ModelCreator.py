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
                nEstimators=100, maxDepth=None, minSampleSplit=2,
                minSampleLeaf=1, maxFeatures="auto", maxLeafNodes=None, maxSamples=None,
                hyperParameters=None, performanceModus=None,
                doHyperParameterisation=False, hyperParameterRange=None, parametersToAddOrOverwrite=None,
                seed=42):
        self.modelType = modelType
        self.setDefaultModelParameters(kernel, gamma, C, maxDepth, maxFeatures,
                                       nEstimators, minSampleLeaf, minSampleSplit,
                                       maxLeafNodes, maxSamples)
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

    def setDefaultModelParameters(self, kernel, gamma, C, maxDepth, maxFeatures,
                                  nEstimators, minSampleLeaf, minSampleSplit,
                                  maxLeafNodes, maxSamples):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.nEstimators = nEstimators # reduces training time
        self.max_depth = maxDepth
        self.min_samples_split = minSampleSplit
        self.min_samples_leaf = minSampleLeaf
        self.max_features = maxFeatures # close to sqrt(nFeatures) generally good
        self.max_leaf_nodes = maxLeafNodes
        self.max_samples = maxSamples # reduces training time

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
            model = self.setDefaultModel()
            model.fit(X_train, y_train)
        if self.modelType == "k-neighbors":
            self.hyperParameters = model.get_params()
            model = model.best_estimator_
        return model, time.time()-startingTime

    def perfromeHyperParameterisation(self, X_train, y_train, parameters=None):
        if parameters is None:
            parameters = self.calcDefaultHyperPar(X_train)
        assert self.parametersToAddOrOverwrite is None or isinstance(self.parametersToAddOrOverwrite, dict), "parametersToAddOrOverwrite needs to be either None or a dictionary, {} != dict. It looks like: {}".format(type(parametersToAddOrOverwrite), parametersToAddOrOverwrite)
        if isinstance(self.parametersToAddOrOverwrite, dict):
            parameters = self.addOrOverwriteParameters(parameters)
        model = self.setDefaultModel()
        model = GridSearchCV(model, parameters, scoring="accuracy", cv=5, n_jobs=100, verbose=1)#, n_jobs=1, cv=10)
        model.fit(X_train, y_train)
        return model, model.best_params_, model.cv_results_

    def calcDefaultHyperPar(self, X_train):
        # defaultGamma = [1/X_train.shape[1], 1/(X_train.shape[1]*np.var(X_train))]
        # print("defaultGamma", defaultGamma)
        # gammaSmall = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100] #np.linspace(0.00000001, 0.0001, 30, endpoint=False)
        gammaSmall = np.linspace(0.00005,0.005,51) #np.linspace(0.00000001, 0.0001, 30, endpoint=False)
        gammaBig = np.linspace(0.008, 0.1, 10)
        # gammaSmall = np.linspace(0.0001, 1, 30, endpoint=False)
        # gammaBig = np.linspace(0.01, 1000, 15)
        # gammaParameter = np.concatenate([gammaSmall, gammaBig, defaultGamma])
        gammaParameter = np.concatenate([np.linspace(0.0001, 0.0002625, 14)]) # every 0.0000125
        cSmall = np.linspace(0.001, 1, 30, endpoint=False)
        cBig = np.linspace(1, 1000, 40, endpoint=False)
        cParameter = np.concatenate([cSmall, cBig])
        cParameter = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5]
        cBig = np.linspace(5, 1000, 5, endpoint=False)
        cParameter = np.concatenate([cParameter, cBig])
        cParameter = np.concatenate([np.arange(100, 981, 75)])
        if self.kernel == "rbf":
            density = 101
            parameters = {'gamma': np.concatenate([self.equallySpacedValueSamplingOverScales([-8,-7], density), self.equallySpacedValueSamplingOverScales([-5, -5], density)]), 'C':np.concatenate([self.equallySpacedValueSamplingOverScales([1,1], density), self.equallySpacedValueSamplingOverScales([4, 4], density)])}
        elif self.kernel == "poly":
            parameters = {"degree": np.arange(3, 9), "gamma":gammaParameter, "C":cParameter}
        else:
            parameters = {"C":cParameter, "gamma":gammaParameter}#{"C":[0.001, 0.1, 1, 5]}#
            #improved sigmoid parameters
            cParameter = np.concatenate([[1,2,3,4,5,10,15,20,30,40,50,55,60,75,100,125,150, 175,200, 204, 250, 300, 350, 400,500,550, 600, 700, 800, 900, 1000, 1005, 1010, 1015]])
            gammaParameter = np.concatenate([np.linspace(0.00001, 0.0001, 18, False), np.linspace(0.001, 0.004, 7)])
            coef0Parameter = np.concatenate([np.linspace(0, 0.5, 31)]) # alternatives would be 25, 31, or 49
            parameters = {"C":cParameter, "gamma":gammaParameter, "coef0":coef0Parameter}
        if self.kernel == "poly" or (self.kernel == "sigmoid" and not "coef0" in parameters):
            parameters["coef0"] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 3, 4, 5, 10]

        # cParameter = np.concatenate([np.linspace(1, 10, 41, False), np.linspace(150, 250, 41, True), [204]])
        # gammaParameter = np.concatenate([np.linspace(0.000001, 0.00001, 41, False), np.linspace(0.00001, 0.0001, 41, False)])
        # parameters = {"C":cParameter, "gamma":gammaParameter}
#         parameters = {'gamma': [5e-05, 0.000149, 0.000248, 0.000347, 0.000446, 0.000545, 0.000644, 0.000743, 0.000842, 0.000941, 0.00104, 0.001139, 0.001238, 0.001337, 0.001436, 0.001535, 0.001634, 0.001733, 0.001832, 0.001931, 0.00203, 0.002129, 0.002228, 0.002327, 0.002426, 0.002525, 0.002624, 0.002723, 0.002822, 0.002921, 0.00302, 0.003119, 0.003218, 0.003317, 0.003416, 0.003515, 0.003614, 0.003713, 0.003812, 0.003911, 0.00401, 0.004109, 0.004208, 0.004307, 0.004406, 0.004505, 0.004604, 0.004703, 0.004802, 0.004901, 0.005, 0.008, 0.0182222222, 0.0284444444, 0.0386666667, 0.0488888889, 0.0591111111, 0.0693333333, 0.0795555556, 0.0897777778, 0.1, 0.0212765957, 0.0226984962],
# 'C': [1e-05, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 204.0, 403.0, 602.0, 801.0]}
        # otherFeatureSetsHv3\diffPar_concat_normPerTissue\svm_k2h_combinedTable_l3f0n2c0ex1
        # parameters = {"C": np.arange(1, 251), "gamma":  np.linspace(1.5e-5, 6.5e-5, 161, False)}
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
            model = svm.SVC(kernel=self.kernel, gamma=self.gamma, C=self.C)
        elif self.modelType == "k-neighbors":
            parameters = {"n_neighbors":np.arange(2, 15)}
            model = GridSearchCV(KNeighborsClassifier(), parameters, cv=3)
        elif self.modelType == "random forest":
            model = RandomForestClassifier(random_state=self.seed,
                                           n_estimators=self.nEstimators,
                                           max_depth=8, #self.max_depth
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           max_features=self.max_features,
                                           max_leaf_nodes=self.max_leaf_nodes,
                                           max_samples=self.max_samples,
                                           n_jobs=-1)
        else:
            model = LogisticRegression(random_state=self.seed, solver='lbfgs', max_iter=20000)
        return model

    def trainModelWithParameters(self, hyperParameters):
        if self.modelType == "svm":
            kernel = hyperParameters["kernel"] if "kernel" in hyperParameters else self.kernel
            if "gamma" in hyperParameters:
                model = svm.SVC(kernel=kernel, gamma=hyperParameters["gamma"], C=hyperParameters["C"])
            else:
                model = svm.SVC(kernel=kernel, C=hyperParameters["C"])
        elif self.modelType == "random forest":
            seed = hyperParameters["seed"] if "seed" in hyperParameters else self.seed
            max_depth = hyperParameters["max_depth"] if "max_depth" in hyperParameters else self.max_depth
            n_estimators = hyperParameters["n_estimators"] if "n_estimators" in hyperParameters else self.n_estimators
            min_samples_split = hyperParameters["min_samples_split"] if "min_samples_split" in hyperParameters else self.min_samples_split
            min_samples_leaf = hyperParameters["min_samples_leaf"] if "min_samples_leaf" in hyperParameters else self.min_samples_leaf
            max_features = hyperParameters["max_features"] if "max_features" in hyperParameters else self.max_features
            max_leaf_nodes = hyperParameters["max_leaf_nodes"] if "max_leaf_nodes" in hyperParameters else self.max_leaf_nodes
            max_samples = hyperParameters["max_samples"] if "max_samples" in hyperParameters else self.max_samples
            model = RandomForestClassifier(random_state=seed,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf,
                                           max_features=max_features,
                                           max_leaf_nodes=max_leaf_nodes,
                                           max_samples=max_samples)
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
            f1Score, accuracy, TPRate, TNRate, auc = MyScorer(y_test, predictions,
                                                         y_scores=y_scores,
                                                         nrOfClasses=self.nrOfClasses
                                                         ).GetAllScores()
            return [f1Score, accuracy, TPRate, TNRate, auc]

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
