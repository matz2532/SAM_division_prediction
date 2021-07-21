import networkx as nx
import numpy as np
import os, sys
sys.path.insert(0, "./Code/")
from CellInSAMCenterDecider import CellInSAMCenterDecider
from utils import convertTextToLabels

class GraphCenterCellFinder (object):

    def __init__(self, baseFolder, plantName, timePoint, centerRadius, graph,
                 peripheralLabelFilename, geometryFilename, centerCellLabels=None,
                 removeSecondPeripheralCells=True):
        self.baseFolder = baseFolder
        self.plantName, self.timePoint = plantName, timePoint
        self.centerRadius = centerRadius
        self.graph = graph
        self.peripheralLabelFilename = peripheralLabelFilename
        self.geometryFilename = geometryFilename
        self.centerCellLabels = centerCellLabels
        self.removeSecondPeripheralCells = removeSecondPeripheralCells

    def defineValidLabels(self):
        centralRegionLabels = self.setCentralRegionLabel()
        peripheralLabels = convertTextToLabels(self.peripheralLabelFilename,
                                allLabelsFilename=self.geometryFilename).GetLabels(onlyUnique=True)
        peripheralLabels = peripheralLabels.astype(int)
        if self.removeSecondPeripheralCells:
            peripheralLabels = self.addNeighboringLabelsOfCurrentGraph(peripheralLabels)
        allowedLabels = self.furtherFilterCentralRegions(centralRegionLabels, peripheralLabels)
        additionalLabelsToRemoveFilename = self.baseFolder + "{}/{}{}T{}.txt".format(self.plantName, "additionalLabelsToRemove", self.plantName, self.timePoint)
        if os.path.isfile(additionalLabelsToRemoveFilename):
            additionalLabelsToRemove = convertTextToLabels(additionalLabelsToRemoveFilename,
                                                allLabelsFilename=self.geometryFilename).GetLabels(onlyUnique=True)
            allowedLabels = self.furtherFilterCentralRegions(allowedLabels, additionalLabelsToRemove)
        return allowedLabels

    def setCentralRegionLabel(self):
        if not self.centerCellLabels is None:
            currentCenterCellLabels = self.centerCellLabels
            myCellInSAMCenterDecider = CellInSAMCenterDecider(self.geometryFilename,
                                        currentCenterCellLabels, centerRadius=self.centerRadius)
            centralRegionLabels = myCellInSAMCenterDecider.GetCentralCells()
        else:
            centralRegionLabels = None
        return centralRegionLabels

    def addNeighboringLabelsOfCurrentGraph(self, peripheralLabels):
        adjacencyList = nx.to_dict_of_dicts(self.graph)
        additionalLabels = []
        for label in peripheralLabels:
            if label in adjacencyList:
                additionalLabels.extend(list(adjacencyList[label].keys()))
        additionalLabels = np.unique(additionalLabels)
        addionalPeripheryCells = np.concatenate([peripheralLabels, additionalLabels])
        return np.unique(addionalPeripheryCells)

    def furtherFilterCentralRegions(self, allowedLabels, labelsToRemove):
        isLabelRemaining = np.isin(allowedLabels, labelsToRemove, invert=True)
        allowedLabels = allowedLabels[isLabelRemaining]
        return allowedLabels

def main():
    myGraphCenterCellFinder = GraphCenterCellFinder()

if __name__ == '__main__':
    main()
