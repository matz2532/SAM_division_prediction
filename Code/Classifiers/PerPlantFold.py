import numpy as np

class PerPlantFold (object):

    def __init__(self, plantNameVector, balanceData=False):
        self.plantNameVector = plantNameVector
        self.balanceData = balanceData

    def split(self, featureMatrix, labelVector):
        assert len(self.plantNameVector) == len(labelVector), "The plant name vector and label vector need to be of the same length, {} != {}".format(len(self.plantNameVector), len(labelVector))
        assert len(labelVector) == len(featureMatrix), "The label vector and feature matrix need to be of the same length, {} != {}".format(len(labelVector), len(featureMatrix))
        trainIndex, testIndex = self.applyPerTissueSplit(featureMatrix, labelVector)
        return zip(trainIndex, testIndex)

    def applyPerTissueSplit(self, featureMatrix, labelVector):
        trainIndex, testIndex = [], []
        uniquePlantNames = self.GetUniquePlantNames()
        for testPlantName in uniquePlantNames:
            isTestPlantName = testPlantName == self.plantNameVector
            currentTestIndex = np.where(isTestPlantName)[0]
            currentTrainIndex = np.where(np.invert(isTestPlantName))[0]
            if self.balanceData:
                currentTestIndex = self.balanceIndicesByLabels(currentTestIndex, labelVector)
                currentTrainIndex = self.balanceIndicesByLabels(currentTrainIndex, labelVector)
            trainIndex.append(currentTrainIndex)
            testIndex.append(currentTestIndex)
        return trainIndex, testIndex

    def balanceIndicesByLabels(self, selectedIndices, labelVector):
        selectedLabels = labelVector[selectedIndices]
        label, counts = np.unique(selectedLabels, return_counts=True)
        if np.all(counts==np.min(counts)):
            return selectedIndices
        nrOfLabels = len(counts)
        minCount = np.min(counts)
        selectedBalanceIndices = np.zeros(minCount*nrOfLabels, dtype=int)
        nrOfSelectedLabels = np.zeros(nrOfLabels)
        j = 0
        for i in range(len(selectedLabels)):
            currentLabel = selectedLabels[i]
            if  nrOfSelectedLabels[currentLabel] < minCount:
                nrOfSelectedLabels[currentLabel] += 1
                selectedBalanceIndices[j] = i
                j += 1
        selectedIndices = selectedIndices[selectedBalanceIndices]
        selectedLabels = labelVector[selectedIndices]
        label, counts = np.unique(selectedLabels, return_counts=True)
        return selectedBalanceIndices

    def GetUniquePlantNames(self):
        _, idx = np.unique(self.plantNameVector, return_index=True)
        return self.plantNameVector[idx]
