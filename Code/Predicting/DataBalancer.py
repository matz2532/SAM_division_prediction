import numpy as np

class DataBalancer (object):

    def __init__(self, features, labels, seed=42):
        self.features = features.copy()
        self.labels = labels.copy()
        self.seed = seed
        np.random.seed(self.seed)
        if not features is None and not labels is None:
            self.indicesForBalancedSet = self.GetIndicesForBalancedSet(self.labels)
            self.balanceFeatures = self.features[self.indicesForBalancedSet, :]
            self.balanceLabels = self.labels[self.indicesForBalancedSet]

    def GetIndicesForBalancedSet(self, labelVector, randomise=False):
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

    def getIdxOfLabels(self, labelVector):
        idxOfUniqueLabels = []
        uniqueLabels = np.unique(labelVector)
        for uniqueLabel in uniqueLabels:
            currentIdxOfUniqueLabels = np.where(labelVector==uniqueLabel)[0]
            idxOfUniqueLabels.append(currentIdxOfUniqueLabels)
        return idxOfUniqueLabels

    def getBalancedIdx(self, idxOfUniqueLabels):
        nrOfLabels = []
        canidateIdx = []
        for i in range(len(idxOfUniqueLabels)):
            nrOfLabels.append(len(idxOfUniqueLabels[i]))
        minNrOfLabels = np.min(nrOfLabels)
        for i in range(len(idxOfUniqueLabels)):
            canidateIdx.append(idxOfUniqueLabels[i][:minNrOfLabels])
        return canidateIdx

    def balanceDataSet(self, seed=42):
        minorityClass = self.determineMinorityClass()
        np.unique()
        np.random.seed(seed)
        np.random.shuffle(nonDividingCellIdx)
        min = np.min(np.unique(self.labels, return_counts=True)[1])
        nonDividingCellIdxToKeep = nonDividingCellIdx[:min]
        for i in nonDividingCellIdxToKeep:
            isToKeep[i] = True
        self.features = self.features[np.where(isToKeep)[0], :]
        self.labels = self.labels[isToKeep]

    def determineMinorityClass(self):
        l, counts = np.unique(self.labels, return_counts=True)
        return l[np.argmin(counts)]

    def balanceDataSet_old(self, seed=42):
        isToKeep = self.labels == 1
        isNonDividingCell = self.labels != 1
        nonDividingCellIdx = np.where(isNonDividingCell)[0]
        np.random.seed(seed)
        np.random.shuffle(nonDividingCellIdx)
        min = np.min(np.unique(self.labels, return_counts=True)[1])
        nonDividingCellIdxToKeep = nonDividingCellIdx[:min]
        for i in nonDividingCellIdxToKeep:
            isToKeep[i] = True
        self.balanceFeatures = self.features[np.where(isToKeep)[0], :]
        self.balanceLabels = self.labels[isToKeep]

    def GetBalancedData(self):
        return self.balanceFeatures, self.balanceLabels

def main():
    X_train = np.load("Temporary/X_train.npy")
    y_train = np.load("Temporary/y_train.npy")
    print(np.unique(y_train, return_counts=True))
    myDataBalancer = DataBalancer(X_train, y_train)
    balancedX, balancedY = myDataBalancer.GetBalancedData()
    print(np.unique(balancedY, return_counts=True))

if __name__ == '__main__':
    main()
