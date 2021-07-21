import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "./Code/DivEventPrediction/")
from TopologyPredictonDataCreator import saveFeatureSets as saveTopoFeatureSets
from DivEventDataCreator import saveFeatureSets as saveDivisionFeatureSets
from TopologyPredictonManager import TopologyPredictonManager

class DividingCellFromFullMappingExtractor (object):

    # convert parent labeling file of all manually mapped cells (lineage tracking in MGX)
    # and exclude the non dividing cells
    def __init__(self, dataBaseFolder):
        self.dataBaseFolder = dataBaseFolder

    def createFeaturesAndLabels(self, centralCellsDict, printCheckCentralCells=True,
                                extractDividingParentLabeling=True):
        plantNames = list(centralCellsDict.keys())
        if printCheckCentralCells:
            print("The central cells dict looks like this:")
            print(centralCellsDict)
        if extractDividingParentLabeling:
            self.saveDividingParentLabelingFromFull(centralCellsDict)

    def saveDividingParentLabelingFromFull(self, centralCells, lengthOfTimeStep=1,
                            fullParentLabelingName="fullParentLabeling",
                            dividingParentLabelingName="parentLabeling"):
        plantNames = list(centralCells.keys())
        for plant in plantNames:
            numberOfTimeSteps = len(centralCells[plant])
            for timePointIdx in range(numberOfTimeSteps):
                isSampleSkipped = len(centralCells[plant][timePointIdx]) == 0
                if not isSampleSkipped:
                    nextTimePoint = timePointIdx + lengthOfTimeStep
                    sampleStepName = "{}T{}T{}".format(plant, timePointIdx, nextTimePoint)
                    folder = "{}{}/".format(self.dataBaseFolder, plant)
                    fullTableName = "{}{}{}.csv".format(folder, fullParentLabelingName, sampleStepName)
                    divCellsParentLabelingDf = self.extractDividingCellsFrom(fullTableName)
                    divTableName = "{}{}{}.csv".format(folder, dividingParentLabelingName, sampleStepName)
                    print(divTableName)
                    divCellsParentLabelingDf.to_csv(divTableName, index=False)

    def extractDividingCellsFrom(self, fullTableName, parentLabelIdx=1):
        fullParentLabelingDf = pd.read_csv(fullTableName)
        parentLabels = fullParentLabelingDf.iloc[:, parentLabelIdx]
        uniqueParentLabels, counts = np.unique(parentLabels, return_counts=True)
        dividingParentLabels = uniqueParentLabels[counts > 1]
        idxOfDivParents = np.where(np.isin(parentLabels, dividingParentLabels))[0]
        divParentLabelingDf = fullParentLabelingDf.iloc[idxOfDivParents, :]
        return divParentLabelingDf

def main():
    dataBaseFolder = "Data/ktn/"
    # each key corresponds to one plant and the columns to the time point
    # you have to give a list [] for each plant and time point even though you may skip this sample
    centralCells = {"ktnP1": [ [], [3839, 3959] ],
                    "ktnP2": [ [23], [424, 426, 50] ],
                    "ktnP3": [ [29, 199,527], [424, 28, 431] ] }
    samplesToSkip = [0] # add indices of samples to skip (in case they aren't done or don't show dividing cells)
                              # the counting starts with 0 and goes from left to right and than top to bottom
    myDividingCellFromFullMappingExtractor = DividingCellFromFullMappingExtractor(dataBaseFolder)
    myDividingCellFromFullMappingExtractor.createFeaturesAndLabels(centralCells)

if __name__ == '__main__':
    main()
