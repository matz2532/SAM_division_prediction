import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "./Code/Feature and Label Creation/")
from PeripheralCellIdentifier import PeripheralCellIdentifier

class NeighborsOfDividingCellMapper (object):

    def __init__(self, parentConnectivityNetwork, daughterConnectivityNetwork,
                 dividingParentDaughterLabeling,fullParentLabeling,
                 dontIncludePeripheralNeighbours=True,
                 plant="unknown", timePoint="unknown"):
        self.SetNetworksAndLabeling(parentConnectivityNetwork, daughterConnectivityNetwork,
                                    dividingParentDaughterLabeling, fullParentLabeling,
                                    plant=plant, timePoint=timePoint)
        self.dontIncludePeripheralNeighbours = dontIncludePeripheralNeighbours
        self.parentNeighboursToExclude = self.calcParentNeighboursToExclude(self.dontIncludePeripheralNeighbours)
        self.mappedCells = self.extractNeighborsOfDividingCells()

    def SetNetworksAndLabeling(self, parentConnectivityNetwork, daughterConnectivityNetwork,
                               dividingParentDaughterLabeling, fullParentLabeling,
                               plant="unknown", timePoint="unknown"):
        self.parentConnectivityNetwork = parentConnectivityNetwork
        self.daughterConnectivityNetwork = daughterConnectivityNetwork
        self.dividingParentDaughterLabeling = dividingParentDaughterLabeling
        self.fullParentLabeling = fullParentLabeling
        self.initaliseParameter(plant, timePoint)

    def initaliseParameter(self, plant, timePoint):
        self.daughterNodeLabels = np.asarray(self.daughterConnectivityNetwork.nodes())
        self.origin = "(parent plant {} time point {})".format(plant, timePoint)
        self.validateCellsExisitingInGeometryNetworkAndParentLabeling()

    def validateCellsExisitingInGeometryNetworkAndParentLabeling(self):
        self.checkExistenceOfDividingParentCellsInNetwork()
        self.expectAllCellsToHaveCoordinates(self.parentConnectivityNetwork, "parent")
        self.expectAllCellsToHaveCoordinates(self.daughterConnectivityNetwork, "daughter")
        self.checkExistenceOfDividedDaughterCellsInNetwork()

    def checkExistenceOfDividingParentCellsInNetwork(self):
        dividingParentCells = np.asarray(list(self.dividingParentDaughterLabeling.keys()))
        allParentCells = self.parentConnectivityNetwork.nodes()
        isDividingParentNotIn = np.isin(dividingParentCells, allParentCells, invert=True)
        assert np.sum(isDividingParentNotIn) == 0, "The dividing cell/s {} is/are not exisiting in the network. {}".format(dividingParentCells[isDividingParentNotIn], self.origin)

    def expectAllCellsToHaveCoordinates(self, graph, name):
        allCellsAndData = np.asarray(list(graph.nodes(data=True)))
        nrOfCells = len(allCellsAndData)
        doesntContainCoordinates = np.full(nrOfCells, False)
        for i in range(nrOfCells):
            if not "coordinates" in allCellsAndData[i][1]:
                doesntContainCoordinates[i] = True
        assert np.sum(doesntContainCoordinates) == 0, "The {} cell/s {} have no coordinates. There is problem with either the network or the geometry file. {}".format(name, allCellsAndData[doesntContainCoordinates], self.origin)

    def checkExistenceOfDividedDaughterCellsInNetwork(self):
        if len(list(self.dividingParentDaughterLabeling.values())) == 0:
            dividedDaughterCells = np.asarray([])
        else:
            dividedDaughterCells = np.concatenate(list(self.dividingParentDaughterLabeling.values()))
        allDaughterCells = self.daughterConnectivityNetwork.nodes()
        isDividedDaughterNotIn = np.isin(dividedDaughterCells, allDaughterCells, invert=True)
        assert np.sum(isDividedDaughterNotIn) == 0, "The divided daughter cells {} do not exisiting in the daughter network. {} Valid cells of the daughter network: {}".format(dividedDaughterCells[isDividedDaughterNotIn], self.origin, np.sort(allDaughterCells))

    def calcParentNeighboursToExclude(self, dontIncludePeripheralNeighbours):
        parentNeighboursToExclude = []
        if dontIncludePeripheralNeighbours:
            peripheralCells = PeripheralCellIdentifier(self.parentConnectivityNetwork).GetPeripheralCells()
            parentNeighboursToExclude.extend(list(peripheralCells))
        return np.asarray(parentNeighboursToExclude)

    def extractNeighborsOfDividingCells(self):
        dividingParentKeys = np.asarray(list(self.dividingParentDaughterLabeling.keys()))
        allParentKeys = np.asarray(list(self.fullParentLabeling.keys()))
        overlappingKeys = allParentKeys[np.isin(allParentKeys, dividingParentKeys)]
        mappedCells = self.mapNeighboursFor(overlappingKeys)
        # print("{} of {} many dividing parent cells are included in full parent labeling".format(len(mappedCells), len(dividingParentKeys)))
        return mappedCells

    def mapNeighboursFor(self, overlappingKeys):
        mapping = {}
        for parentCell in overlappingKeys:
            mappedDaughterNeighbors = self.mappNeighboursOf(parentCell)
            if len(mappedDaughterNeighbors) > 0:
                mapping[parentCell] = mappedDaughterNeighbors
        return mapping

    def mappNeighboursOf(self, parentCell, printOutNotIncludedNeighbours=False):
        mappedDaughterNeighbors = {}
        parentNeighbors = np.asarray(list(self.parentConnectivityNetwork.neighbors(parentCell)))
        isNeighborMappable = [n in self.fullParentLabeling for n in parentNeighbors]
        mappableParentNeighbor = parentNeighbors[isNeighborMappable]
        isNeighbourAllowed = np.isin(mappableParentNeighbor, self.parentNeighboursToExclude, invert=True)
        mappableParentNeighbor = mappableParentNeighbor[isNeighbourAllowed]
        for n in mappableParentNeighbor:
            mappedCells = self.fullParentLabeling[n]
            mappedDaughterNeighbors[n] = mappedCells[0]
        if printOutNotIncludedNeighbours and np.any(np.invert(isNeighborMappable)):
            notMappedNeighbours = parentNeighbors[np.invert(isNeighborMappable)]
            print("The neighbours {} of parent {} {} are not mappable as they are not inside the fullParentLabeling table.".format(notMappedNeighbours, parentCell, self.origin))
        return mappedDaughterNeighbors

    def GetMappedCells(self):
        return self.mappedCells

def convertParentLabelingTableToDict(parentLabelingTable):
    parentLabelingDict = {}
    nrOfLabels = parentLabelingTable.shape[0]
    for i in range(nrOfLabels):
        daughter, parent = parentLabelingTable.iloc[i, 0], parentLabelingTable.iloc[i, 1]
        if parent in parentLabelingDict:
            parentLabelingDict[parent].append(daughter)
        else:
            parentLabelingDict[parent] = [daughter]
    return parentLabelingDict

def compareParentLabeling(newParentLabeling, oldParentLabeling):
    # are all keys included in the other
    newKeys = np.asarray(list(newParentLabeling.keys()))
    oldKeys = np.asarray(list(oldParentLabeling.keys()))
    isOldKeyInNewKey = np.isin(newKeys, oldKeys)
    isNewKeyInOldKey = np.isin(oldKeys, newKeys)
    print(np.sum(isOldKeyInNewKey), len(newKeys), newKeys[np.invert(isOldKeyInNewKey)])
    print(np.sum(isNewKeyInOldKey), len(oldKeys), oldKeys[np.invert(isNewKeyInOldKey)])

def main():
    import sys
    from TopologyPredictonDataCreator import TopologyPredictonDataCreator
    dataFolder = "Data/WT/"
    centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    plantNames = ["P1", "P2", "P5", "P6", "P8"]
    plantIdx = 0
    plantName = plantNames[plantIdx]
    myTopologyPredictonDataCreator = TopologyPredictonDataCreator(dataFolder, 5, [plantName], centralCellsDict=centralCellsDict, skipFooterOfGeometryFile=4)
    data = myTopologyPredictonDataCreator.GetData()
    timeIdx = 1
    parentConnectivityNetwork = data[plantName]["graphCreators"][timeIdx].GetGraph()
    daughterConnectivityNetwork = data[plantName]["graphCreators"][timeIdx+1].GetGraph()
    dividingParentDaughterLabeling = data[plantName]["parentDaugherCellLabeling"][timeIdx]
    # fullParentLabeling = pd.read_csv("Data/topoPredData/parentLabeling/fullParentLabeling{}T{}T{}.csv".format(plantName, timeIdx, timeIdx+1))
    # fullParentLabeling = convertParentLabelingTableToDict(fullParentLabeling)
    fullParentLabeling = data[plantName]["fullParentLabeling"][timeIdx]
    # fullParentLabeling = convertParentLabelingTableToDict(fullParentLabeling)
    # compareParentLabeling(dividingParentDaughterLabeling, fullParentLabeling)
    # print(dividingParentDaughterLabeling)
    # print(fullParentLabeling)
    myNeighborsOfDividingCellMapper = NeighborsOfDividingCellMapper(parentConnectivityNetwork,
                                                daughterConnectivityNetwork,
                                                dividingParentDaughterLabeling,
                                                fullParentLabeling)
    mappedCells = myNeighborsOfDividingCellMapper.GetMappedCells()
    print(mappedCells)

if __name__ == '__main__':
    main()
