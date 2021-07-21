import numpy as np
import pandas as pd
import sys
modulePath = "./Code/"
sys.path.insert(0, modulePath)
from utils import convertTextToLabels

class LabelCreator (object):

    def __init__(self, positiveLabelsFilename, featureMatrixFilename,
                allowedLabels=None, areParametersFilenames=[True,True],
                useDividingCells=None):
        self.positiveLabelsFilename = positiveLabelsFilename
        self.useDividingCells = useDividingCells
        self.positiveLabels = self.getPositiveLabels(self.positiveLabelsFilename, isFilename=areParametersFilenames[0])
        if not self.useDividingCells is None:
            self.addPositiveLabels(self.useDividingCells)
        self.allLabels = self.getAllLabes(featureMatrixFilename, isFilename=areParametersFilenames[1])
        if not allowedLabels is None:
            self.positiveLabels = self.removeNotAllowedLabels(self.positiveLabels, allowedLabels)
            self.allLabels = self.removeNotAllowedLabels(self.allLabels, allowedLabels)
        self.labelVector = self.createLabelVector(self.allLabels, self.positiveLabels)
        self.labelVector = pd.DataFrame(self.labelVector)
        self.labelVector.index = self.allLabels

    def getPositiveLabels(self, tableFilename, isFilename=True):
        if isFilename:
            table = pd.read_csv(tableFilename, sep=",")
        else:
            table = tableFilename
        nonUniquePositiveLabels = np.array(table.iloc[:,1])
        uniquePositiveLabels = np.unique(nonUniquePositiveLabels)
        return uniquePositiveLabels

    def getAllLabes(self, tableFilename, isFilename=True):
        if isFilename:
            table = pd.read_csv(tableFilename, sep=",", index_col=0)
        else:
            table = tableFilename
        allLabels = np.array(table.index)
        return allLabels

    def removeNotAllowedLabels(self, possibleLabels, allowedLabels):
        keepLabels = np.isin(possibleLabels, allowedLabels)
        return possibleLabels[keepLabels]

    def createLabelVector(self, allLabels, positiveLabels):
        labelVector = np.isin(allLabels, positiveLabels)
        labelVector = labelVector.astype(int)
        labelVector[labelVector==0] = -1
        return labelVector

    def addPositiveLabels(self, positiveLabelsFilename):
        additionalPositiveLabels = convertTextToLabels(positiveLabelsFilename).GetLabels(onlyUnique=True)
        additionalPositiveLabels = additionalPositiveLabels.astype(int)
        additionalPositiveLabels[np.isin(additionalPositiveLabels, self.positiveLabels, invert=True)]
        self.positiveLabels = np.concatenate([self.positiveLabels, additionalPositiveLabels])

    def GetLabelVector(self):
        return self.labelVector

    def SaveLabelVector(self, filename, sep=","):
        self.labelVector.to_csv(filename, sep=sep)

def main():
    from FeatureVectorCreator import FeatureVectorCreator
    from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
    from CellInSAMCenterDecider import CellInSAMCenterDecider
    folder = "./Data/connectivityNetworks/"
    connectivityNetworkFilename =  "cellularConnectivityNetworkP1T0.csv"
    connectivityGraph = GraphCreatorFromAdjacencyList(folder + connectivityNetworkFilename).GetGraph()
    geometryFilenames = "areaP1T0.csv"
    centerCellLabels = [618, 467, 570]
    myCellInSAMCenterDecider = CellInSAMCenterDecider(folder+geometryFilenames, centerCellLabels, centerRadius=30)
    centralRegionLabels = myCellInSAMCenterDecider.GetCentralCells()
    featureMatrix = FeatureVectorCreator(connectivityGraph).GetFeatureMatrixDataFrame()
    positiveLabelsFilename = "parentLabelingP1T0T1.csv"
    myLabelCreator = LabelCreator(folder + positiveLabelsFilename, featureMatrix, allowedLabels=centralRegionLabels, areParametersFilenames=[True, False])
    print(myLabelCreator.GetLabelVector())
    #myLabelCreator.SaveLabelVector("./Data/connectivityNetworks/LabelVectorP1T0T1.csv")

if __name__ == '__main__':
    main()
