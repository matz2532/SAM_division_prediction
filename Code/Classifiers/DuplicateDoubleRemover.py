import numpy as np
import pandas as pd

class DuplicateDoubleRemover (object):

    def __init__(self, table):
        self.doublesReducedTable = table.copy()
        self.table = table
        self.removeDuplicateFeatures()

    def removeDuplicateFeatures(self):
        self.duplicateColumn = []
        self.duplicateColIdx = []
        self.idxToDuplicateIdxDict = {}
        columnNames = list(self.table.columns)
        for i in range(self.table.shape[1]-1):
            for j in range(i+1, self.table.shape[1]):
                if self.areColumnsTheSame(self.table.iloc[:, i], self.table.iloc[:, j]):
                    if i in self.idxToDuplicateIdxDict:
                        self.idxToDuplicateIdxDict[i].append(j)
                    else:
                        self.idxToDuplicateIdxDict[i] = [j]
                    self.duplicateColumn.append(columnNames[j])
                    self.duplicateColIdx.append(j)
        self.duplicateColumn = np.unique(self.duplicateColumn)
        self.duplicateColIdx = np.unique(self.duplicateColIdx)
        self.doublesReducedTable.drop(self.duplicateColumn, axis="columns", inplace=True)

    def areColumnsTheSame(self, column1, column2):
        return np.all(column1 == column2)

    def GetDoublesReducedTable(self):
        return self.doublesReducedTable

    def GetDuplicateColumn(self):
        return self.duplicateColumn

    def GetDuplicateColIdx(self):
        return self.duplicateColIdx

    def GetNonDuplicateColIdx(self):
        nrOfCols = self.table.shape[1]
        nonDuplicateCols = np.arange(nrOfCols)
        isNonDuplicateCol = np.isin(nonDuplicateCols, self.duplicateColIdx, invert=True)
        nonDuplicateCols = nonDuplicateCols[np.where(isNonDuplicateCol)[0]]
        return nonDuplicateCols

    def GetDuplicateOfIdx(self):
        return self.idxToDuplicateIdxDict

def main():
    tableName = "Data/WT/topoPredData/biologicalFeatures/biologicalAndDiffNetwork_notnormalised.csv"
    tableName = "Data/WT/divEventData/manualCentres/topology/combinedFeatures_topology_notnormalised.csv"
    divSampleData = pd.read_csv(tableName)
    print(divSampleData.shape)
    myDuplicateDoubleRemover = DuplicateDoubleRemover(divSampleData)
    divSampleData = myDuplicateDoubleRemover.GetDoublesReducedTable()
    print(divSampleData.shape)
    columns = list(divSampleData.columns)
    print(columns)
    print(len(columns)-3)

if __name__ == '__main__':
    main()
