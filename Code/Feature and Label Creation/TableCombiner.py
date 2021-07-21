import numpy as np
import pandas as pd

class TableCombiner:

    def __init__(self, data, isParameterFilename=True):
        if isParameterFilename:
            self.data = self.setupData(data)
        else:
            self.data = data
        self.mergedData = pd.concat(self.data)

    def setupData(self, data):
        workingData = []
        for f in data:
            if type(f) == str:
                currentData = pd.read_csv(f)
                workingData.append(currentData)
        return workingData

    def GetMergedData(self):
        return self.mergedData

    def SaveMergedData(self, filename, sep=","):
        self.mergedData.to_csv(filename, sep=sep)

def main():
    testSets = ["P1T0", "P1T1","P1T2", "P1T3"]
    filenames = []
    for set in testSets:
        filenames.append("./Data/connectivityNetworks/featureMatrixExampleGraph{}.csv".format(set))
    myTableCombiner = TableCombiner(filenames)
    myTableCombiner.SaveMergedData("./Data/connectivityNetworks/combinedFeatureMatrix.csv")

if __name__ == '__main__':
    main()
