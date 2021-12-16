import numpy as np

class PerPlantFold (object):

    def __init__(self, plantNameVector, balanceData=False):
        self.plantNameVector = plantNameVector
        self.balanceData = balanceData

    def split(self, featureMatrix, labelVector):
        assert len(self.plantNameVector) == len(labelVector), "The plant name vector and label vector need to be of the same length, {} != {}".format(len(self.plantNameVector), len(labelVector))
        assert len(labelVector) == len(featureMatrix), "The label vector and feature matrix need to be of the same length, {} != {}".format(len(labelVector), len(featureMatrix))
        if self.balanceData:
            trainIndex, testIndex = self.applyBalancedPerTissueSplit(featureMatrix, labelVector)
        else:
            trainIndex, testIndex = self.applyPerTissueSplit(featureMatrix, labelVector)
        return zip(trainIndex, testIndex)

    def applyBalancedPerTissueSplit(self, featureMatrix, labelVector):
        raise NotImplementedError

    def applyPerTissueSplit(self, featureMatrix, labelVector):
        trainIndex, testIndex = [], []
        uniquePlantNames = self.GetUniquePlantNames()
        for testPlantName in uniquePlantNames:
            isTestPlantName = testPlantName == self.plantNameVector
            currentTestIndex = np.where(isTestPlantName)[0]
            currentTrainIndex = np.where(np.invert(isTestPlantName))[0]
            trainIndex.append(currentTrainIndex)
            testIndex.append(currentTestIndex)
        return trainIndex, testIndex

    def GetUniquePlantNames(self):
        _, idx = np.unique(self.plantNameVector, return_index=True)
        return self.plantNameVector[idx]
