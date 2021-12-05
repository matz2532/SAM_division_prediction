import numpy as np
import sys
sys.path.insert(0, "./Code/Classifiers/")
from MyScorer import MyScorer

class ModelEnsambleUtiliser (object):

    def __init__(self, modelList, performanceModus="all performances 1D list", nrOfClasses=2):
        self.performanceModus = performanceModus
        self.nrOfClasses = nrOfClasses
        self.SetModelList(modelList)

    def predict(self, X):
        self.predictedEnsambleValues = self.predictIndividualModels(X)
        self.predictedValues = self.predictMajorityVotedValue()
        return self.predictedValues

    def predictIndividualModels(self, X):
        nrOfSamples = len(X)
        predictedEnsambleValues = np.zeros((nrOfSamples, len(self.modelList)))
        for i, model in enumerate(self.modelList):
            y = model.predict(X)
            predictedEnsambleValues[:, i] = y
        return predictedEnsambleValues

    def predictMajorityVotedValue(self):
        self.tiesBetweenCount = {"0, 1":0,"0, 2":0,"1, 2":0, "1, 0":0,"2, 0":0,"2, 1":0}
        nrOfSamples = len(self.predictedEnsambleValues)
        nrOfTiedValues = 0
        predictedValues = np.zeros(nrOfSamples)
        for sampleIdx in np.arange(nrOfSamples):
            predictedSampleValues = self.predictedEnsambleValues[sampleIdx, :]
            yValues, counts = np.unique(predictedSampleValues, return_counts=True)
            idxOfYPredictedByMostModels = np.argmax(counts)
            if self.isCountTied(counts):
                # print("sample is tied, sampleIdx", sampleIdx, yValues, counts)
                # print(idxOfYPredictedByMostModels)
                max = np.max(counts)
                tiedValues = yValues[counts==max]
                tieLabel1 = tiedValues[0]
                tieLabel2 = tiedValues[1]
                key = "{}, {}".format(int(tieLabel1), int(tieLabel2))
                if key in self.tiesBetweenCount:
                    self.tiesBetweenCount[key] += 1
                else:
                    print("sample is tied with more than 2, sampleIdx", sampleIdx, yValues, counts)
                    print(idxOfYPredictedByMostModels)
                nrOfTiedValues += 1
            else:
                predictedValues[sampleIdx] = yValues[idxOfYPredictedByMostModels]
        return predictedValues

    def isCountTied(self, counts):
        max = np.max(counts)
        isCountTied = counts == max
        if np.sum(isCountTied) > 1:
            return True
        return False

    def decision_function(self, X):
        allDecision = []
        for model in self.modelList:
            decision = model.decision_function(X)
            allDecision.appen(decision)
        self.y_scores = np.mean(allDecision, axis=0)
        return self.y_scores

    def SetModelList(self, modelList):
        self.modelList = modelList
        assert isinstance(self.modelList, list), "The modelList needs to be a list. {} != list".format(self.modelList)
        # ensure that all model have predict function
        self.predictedEnsambleValues = None
        self.predictedValues = None

    def GetModelList(self):
        return self.modelList

    def GetModel(self):
        return self.modelList

    def GetAllHyperParameters(self):
        allHyperParameters = []
        for model in self.modelList:
            allHyperParameters.append(model.GetAllHyperParameters())
        return allHyperParameters

    def TestModel(self, X_test, y_test, model=None):
        print("ensamble")
        predictedValues = self.predict(X_test)
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
            print("returning performances", performances)
            return performances

    def PrintNrOfTiesBetweenCount(self, additionalPrint="", printAlways=False):
        if np.any([i > 0 for i in self.tiesBetweenCount.values()]) or printAlways:
            print(additionalPrint, "tiesBetweenCount", self.tiesBetweenCount)

    def decision_function(self, X):
        decisionScores = []
        for model in self.modelList:
            assert hasattr(model, "decision_function"), "The model {} has no decision_function."
            decisionScore = model.decision_function(X)
            decisionScores.append(decisionScore)
        meanDecisionScore = np.zeros_like(decisionScore)
        for decisionScore in decisionScores:
            meanDecisionScore += decisionScore
        meanDecisionScore /= len(decisionScores)
        return meanDecisionScore

def main():
    import pickle
    import sys
    sys.path.insert(0, "./Code/Classifiers/")

    from MyScorer import MyScorer

    dataFolder = "Results/Tissue mapping/testingEnsambleModel/"
    modelFolder = "Results/Tissue mapping/{}/models.pkl"
    modelNames = ["svm_k2_combinedTable_l3f2n1c0ex1",
                #  "random forest_h_combinedTable_l3f2n1c0ex1",
                  "svm_k4h_combinedTable_l3f2n1c0ex1"]
    allModelFilenames = [modelFolder.format(nodelName) for nodelName in modelNames]
    valXs = pickle.load(open(dataFolder+"combinedTable_l3f2n1c0ex1_valXs.pkl", "rb"))
    valYs = pickle.load(open(dataFolder+"combinedTable_l3f2n1c0ex1_valYs.pkl", "rb"))
    allModels = []
    for modelFilename in allModelFilenames:
        model = pickle.load(open(modelFilename, "rb"))
        allModels.append(model)
    allAcc = []
    for split in range(len(valXs)):
        modelList = []
        for modelsOfSplits in allModels:
            modelList.append(modelsOfSplits[split])
        currentValX = valXs[split]
        currentValY = valYs[split]
        myModelEnsambleUtiliser = ModelEnsambleUtiliser(modelList)
        predictedScoreValY = myModelEnsambleUtiliser.decision_function(currentValX)
        print(predictedScoreValY[:3, :])
        sys.exit()
        predictedValY = myModelEnsambleUtiliser.predict(currentValX)
        ensambleAcc = MyScorer().calcAccuracy(currentValY, predictedValY)
        print("split:", split, "Acc:", ensambleAcc)
        # pickle.dump(predictedValY, open("{}predicted split {}.pkl".format(dataFolder, split), "wb"))
        allAcc.append(ensambleAcc)
    print("ensamble acc: {}+-{}".format(np.round(np.mean(allAcc), 5), np.round(np.std(allAcc), 5)))
    accOfSigmoidSvmH = [55.9, 59.9, 61.6, 63.5, 57.1]
    print("sigmoid svm h acc: {}+-{}".format(np.round(np.mean(accOfSigmoidSvmH), 5), np.round(np.std(accOfSigmoidSvmH), 5)))

if __name__ == '__main__':
    main()
