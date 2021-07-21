import numpy as np
import pandas as pd

class DividingCellInTableIdentifier (object):

    def __init__(self, table, parentLabelling=None,
                 plantNameColumn=0, timePointColumn=1,
                 parentLabelColumn=1, sep=","):
        self.SetTable(table)
        self.plantNameColumn = plantNameColumn
        self.timePointColumn = timePointColumn
        self.parentLabelColumn = parentLabelColumn
        self.sep = sep
        if type(parentLabelling) == dict:
            self.calcDividingCells(parentLabelling)

    def calcDividingCells(self, parentLabelling):
        assert type(parentLabelling) == dict, "parentLabelling needs to be a dictionary containing the plant names (as keys) used in the table and a dict containing the time points (as keys) with the actual parent labelling table or its filename to the csv."
        allDividingCellIdxOfTissue = []
        for plantName, labellingOfTimePoints in parentLabelling.items():
            assert isinstance(labellingOfTimePoints, dict), "parentLabelling needs to be a dictionary containing the plant names (as keys) used in the table and a dict containing the time points (as keys) with the actual parent labelling table or its filename to the csv."
            for timePoint, currentLabelling in labellingOfTimePoints.items():
                assert isinstance(currentLabelling, pd.core.frame.DataFrame) or isinstance(currentLabelling, str) or isinstance(currentLabelling, dict), "The current labelling file/filename from {} {} needs of type pandas.core.frame.DataFrame, string, or dict, but is {}".format(plantName, timePoint, type(currentLabelling))
                if isinstance(currentLabelling, dict):
                    dividingParents = list(currentLabelling.keys())
                else:
                    if isinstance(currentLabelling, str):
                        currentLabelling = pd.read_csv(currentLabelling, sep=self.sep)
                    dividingParents = self.determineDividingParents(currentLabelling)
                indicesToRemove = self.findIdxOfDividingParents(plantName,
                                                                timePoint,
                                                                dividingParents)
                allDividingCellIdxOfTissue.append(indicesToRemove)
        self.allDividingCellIdx = np.concatenate(allDividingCellIdxOfTissue).astype(int)
        return self.allDividingCellIdx

    def determineDividingParents(self, currentLabellingTable):
        parentLabels = currentLabellingTable.iloc[:, self.parentLabelColumn].to_numpy()
        labels, counts = np.unique(parentLabels, return_counts=True)
        dividingParents = labels[counts > 1]
        return dividingParents

    def findIdxOfDividingParents(self, plantName, timePoint, dividingParents):
        isCellOfTissue = self.isCellOfTissue(plantName, timePoint)
        whereIsCellOfTissue = np.where(isCellOfTissue)[0]
        idxOfDividingCellInTable = []
        for i, cell in enumerate(self.table.iloc[whereIsCellOfTissue, 3]):
            if cell in dividingParents:
                idxOfDividingCellInTable.append(whereIsCellOfTissue[i])
        return idxOfDividingCellInTable

    def isCellOfTissue(self, plantName, timePoint):
        isCorrectPlant = np.equal(self.table.iloc[:, self.plantNameColumn], plantName)
        isCorrectTime = np.equal(self.table.iloc[:, self.timePointColumn], timePoint)
        return isCorrectPlant & isCorrectTime

    def SetTable(self, table):
        self.table = table
        assert isinstance(self.table, pd.core.frame.DataFrame), "The table needs of type pandas.core.frame.DataFrame, but is {}".format(type(self.table))

    def GetAllDividingCellIdxInTable(self):
        return self.allDividingCellIdx

def createExcludeDividingNeighboursDict(dataFolder):
    excludeDividingNeighboursDict = {}
    for plantName in ["P1", "P2", "P5", "P6", "P8"]:
        dividingCellsPerTimePoint = {}
        for timePoint in np.arange(5-1):
            parentDaughterLabellingDict = pd.read_csv("{}{}/parentLabeling{}T{}T{}.csv".format(dataFolder, plantName, plantName, timePoint, timePoint+1))
            dividingCellsPerTimePoint[timePoint] = parentDaughterLabellingDict
        excludeDividingNeighboursDict[plantName] = dividingCellsPerTimePoint
    return excludeDividingNeighboursDict

def main():
    tableFilenames = "Data/WT/topoPredData/diff/manualCentres/allTopos/combinedFeatures_allTopos_notnormalised.csv"
    parentLabellingFilename = "Data/WT/P1/parentLabelingP1T0T1.csv"
    table = pd.read_csv(tableFilenames)
    parentLabelling = pd.read_csv(parentLabellingFilename)
    allParentLabellings = createExcludeDividingNeighboursDict("Data/WT/")# {"P1":{0:parentLabelling}}
    print(allParentLabellings)
    myDividingCellInTableIdentifier = DividingCellInTableIdentifier(table,
                                        allParentLabellings)
    indices = myDividingCellInTableIdentifier.GetAllDividingCellIdxInTable()
    print(indices)

if __name__ == '__main__':
    main()
