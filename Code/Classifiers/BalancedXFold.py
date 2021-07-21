import numpy as np
from sklearn.model_selection import KFold

class BalancedXFold (object):

    def __init__(self, n_splits=0, random_state=None): #use **args
        self.n_splits = n_splits
        self.random_state = random_state
        np.random.seed(seed=self.random_state)

    def split(self, featureMatrix, labelVector):
        self.uniqueLabels, self.frequencyOfUniqueLabels = np.unique(labelVector,return_counts=True)
        # if self.areLablesUnbalanced(labelVector, percentageDeviation=0.1):
        idxOfUniqueLabels  = self.getIdxOfLabels(labelVector)
        train_index, test_index = self.applyBalancedSplit(idxOfUniqueLabels, featureMatrix, labelVector)
        return zip(train_index, test_index)

    def areLablesUnbalanced(self, labelVector, percentageDeviation=0.1):
        relativeFrequency = self.frequencyOfUniqueLabels/np.sum(self.frequencyOfUniqueLabels)
        expectedFrequency = np.full(len(self.uniqueLabels), 1/len(self.uniqueLabels))
        deviations = expectedFrequency-relativeFrequency
        return np.max(np.abs(deviations)) > percentageDeviation

    def getIdxOfLabels(self, labelVector):
        idxOfUniqueLabels = []
        uniqueLabels = np.unique(labelVector)
        for uniqueLabel in uniqueLabels:
            currentIdxOfUniqueLabels = np.where(labelVector==uniqueLabel)[0]
            idxOfUniqueLabels.append(currentIdxOfUniqueLabels)
        return idxOfUniqueLabels

    def applyBalancedSplit(self, idxOfUniqueLabels, featureMatrix, labelVector):
        kFold = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for i in range(len(idxOfUniqueLabels)):
            np.random.shuffle(idxOfUniqueLabels[i])
        canidates = self.getBalancedIdx(idxOfUniqueLabels)
        train_index, test_index = self.createEmptyTrainTestIndices(self.n_splits)
        for i in range(len(canidates)):
            currentSplit = 0
            for tmpTrainIndex, tmpTestIndex in kFold.split(canidates[i]):
                train_index[currentSplit].extend(list(canidates[i][tmpTrainIndex]))
                test_index[currentSplit].extend(list(canidates[i][tmpTestIndex]))
                currentSplit += 1
        for i in range(self.n_splits):
            np.random.shuffle(train_index[i])
            np.random.shuffle(test_index[i])
        return train_index, test_index

    def getBalancedIdx(self, idxOfUniqueLabels):
        nrOfLabels = []
        canidateIdx = []
        for i in range(len(idxOfUniqueLabels)):
            nrOfLabels.append(len(idxOfUniqueLabels[i]))
        minNrOfLabels = np.min(nrOfLabels)
        for i in range(len(idxOfUniqueLabels)):
            canidateIdx.append(idxOfUniqueLabels[i][:minNrOfLabels])
        return canidateIdx

    def createEmptyTrainTestIndices(self, n_splits):
        train_index, test_index = [], []
        for i in range(n_splits):
            train_index.append([])
            test_index.append([])
        return train_index, test_index

    def GetBalancedSet(self, labelVector, randomise=False):
        idxOfUniqueLabels = self.getIdxOfLabels(labelVector)
        if randomise is True:
            for i in range(len(idxOfUniqueLabels)):
                np.random.shuffle(idxOfUniqueLabels[i])
        downSampledIdxLabels = self.getBalancedIdx(idxOfUniqueLabels)
        balancedSet = []
        for labelIdx in downSampledIdxLabels:
            balancedSet.extend(list(labelIdx))
        np.random.shuffle(balancedSet)
        return balancedSet

def main():
    import pandas as pd
    from SVMCreator import SVMCreator
    featureMatrixFilename = "./Data/connectivityNetworks/combinedFeatures.csv"
    featureMatrix = pd.read_csv(featureMatrixFilename, sep=",", index_col=0)
    labelVector =  pd.read_csv("./Data/connectivityNetworks/combinedLabels.csv", sep=",", index_col=0)
    featureMatrix.fillna(0,inplace=True)
    featureMatrix = featureMatrix.to_numpy(copy=True)
    labelVector = labelVector.to_numpy(copy=True)
    labelVector = labelVector.flatten()
    balancedSet = BalancedXFold(random_state=0).GetBalancedSet(labelVector, randomise=True)
    model = SVMCreator(X_train=featureMatrix[balancedSet], y_train=labelVector[balancedSet]).GetModel()
    print(model.predict(featureMatrix[balancedSet]))
    # myBalancedXFold = BalancedXFold(n_splits=3, random_state=42)
    # for train_index, test_index in myBalancedXFold.split(featureMatrix, labelVector):
    #     print(np.unique(labelVector[train_index],return_counts=True))
    #     print(np.unique(labelVector[test_index],return_counts=True))
    #     print(labelVector[train_index])

if __name__ == '__main__':
    main()
