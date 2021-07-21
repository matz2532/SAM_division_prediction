import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.insert(0, "./Code/Feature and Label Creation/")

from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList

class CentralPositionFinder (object):

    def __init__(self, table, isNetworkTable=False,
                 columns=['Center_X', 'Center_Y', 'Center_Z'],
                 printResults=True, showCoordDistribution=False):
        self.table = table
        self.columns = columns
        self.measure = "mean"
        if isNetworkTable:
            self.centralCells = self.calcCentralCellUsingNetwork(printResults=printResults)
        else:
            self.centralCells = self.calcCentralCellUsingPositions(measure=self.measure,
                                        printResults=printResults)
            # could include multiple cells to be central in case they are closer to centrum than to all other selected cells
            if showCoordDistribution:
                self.plotXYZCoordDistribution()

    def calcCentralCellUsingNetwork(self, printResults=False):
        network = GraphCreatorFromAdjacencyList(self.table).GetGraph()
        eccentricities = nx.eccentricity(network)
        cellLabels = np.asarray(list(eccentricities.keys()))
        eccentricities = list(eccentricities.values())
        isMin = np.isin(eccentricities, np.min(eccentricities))
        leastEccentricCells = cellLabels[isMin]
        if printResults:
            labels, counts = np.unique(eccentricities, return_counts=True)
            print("eccentricity", labels)
            print("counts      ", counts)
            print("least eccentric cells:", leastEccentricCells)
        if isinstance(leastEccentricCells, int):
            leastEccentricCells = [leastEccentricCells]
        return leastEccentricCells

    def calcCentralCellUsingPositions(self, measure="mean", printResults=False,
                                      columnsToPrint=[0,3,4,5]):
        self.positions = self.table[self.columns].to_numpy()
        if measure == "mean":
            mean = np.mean(self.positions, axis=0)
        elif measure == "median":
            mean = np.median(self.positions, axis=0)
        else:
            print("The measure {} is not yet implemented. Only mean and median are allowed".format(measure))
            sys.exit()
        mean[2] = np.max(self.positions[:, 2]) # can also be removed and only slighly reduces performance
        distMean = self.calcDistance(mean, self.positions)
        idxOfBest = np.argsort(distMean)
        if printResults:
            print(measure, mean)
            selectedIdx = [0,1,2]
            print(distMean[idxOfBest[selectedIdx]])
            print(self.table.iloc[idxOfBest[selectedIdx], columnsToPrint])
        return self.table.iloc[idxOfBest, 0].to_numpy()

    def calcDistance(self, fromPoint, toPoints):
        coordDist = toPoints - fromPoint
        eucDist = np.linalg.norm(coordDist, axis=1)
        return eucDist

    def plotXYZCoordDistribution(self):
        plt.subplots_adjust(hspace=0.7)
        nrOfSubPlots = len(self.columns)
        for i in range(nrOfSubPlots):
            plt.subplot(nrOfSubPlots, 1, i+1)
            plt.ylabel("Density")
            sns.kdeplot(self.positions[:, i], shade=True)
            # plt.title(self.columns[i])
            plt.xlabel("Coordinates of "+self.columns[i])
        plt.show()

    def GetCentralCell(self, measure=None, selectedRange=3):
        if measure is None or measure == self.measure:
            centralCells = self.centralCells
        else:
            centralCells = self.calcCentralCell(measure=measure)
        selectedRange = self.ensureAppropriateRange(centralCells, selectedRange)
        return centralCells[selectedRange]

    def ensureAppropriateRange(self, centralCells, selectedRange):
        lenOfCentralCells = len(centralCells)
        if selectedRange == "max":
            selectedRange = np.arange(len(centralCells))
        elif isinstance(selectedRange, int):
            if lenOfCentralCells <= selectedRange:
                print("Your selected range of {} is to big as the length of the central cells is only {}. It will be scaled down to {}".format(selectedRange, lenOfCentralCells, lenOfCentralCells-1))
                selectedRange = lenOfCentralCells - 1
                print("The range is now:", selectedRange)
        else:
            isToBig = selectedRange > lenOfCentralCells-1
            if np.any(isToBig):
                print("The selected range values {} are to big and are removed. The length of the central cells is only {}.".format(selectedRange[isToBig], lenOfCentralCells))
                selectedRange = selectedRange[np.invert(isToBig)]
                print("The range is now:", selectedRange)
        return selectedRange

    def SaveCentralCellAsHeatMap(self, filenameToSave, originalFilename,
                                 selectedRange=0, sep=",", skippedFooter=4,
                                 manualCentralCell=None, sameValue=False,
                                 labelColumn="Label", valueColumn="Value"):
        if isinstance(self.table, str):
            centralCellTable = pd.read_csv(originalFilename)#, skipfooter=skipFooter, engine="python")
            centralCellTable = centralCellTable.iloc[:-2, :]
        else:
            centralCellTable = self.table.copy()
        assert labelColumn in list(centralCellTable.columns), "The column {} is missing as a column. {} not in {}".format(labelColumn, labelColumn, list(centralCellTable.columns))
        assert valueColumn in list(centralCellTable.columns), "The column {} is missing as a column. {} not in {}".format(valueColumn, valueColumn, list(centralCellTable.columns))
        cellLabels = centralCellTable[labelColumn].to_numpy()
        centralCellValue = self.determineCentralCellValues(cellLabels, selectedRange,
                                                           manualCentralCell, sameValue)
        centralCellTable[valueColumn] = centralCellValue
        centralCellTable.to_csv(filenameToSave, sep=sep, index=False)
        if not originalFilename is None:
            lastLines = self.extractLastLines(originalFilename, extractFooter=skippedFooter)
            self.appendLinesToFile(lastLines, filenameToSave)

    def determineCentralCellValues(self, cellLabels, selectedRange="max",
                                   manualCentralCell=None, sameValue=False):
        if manualCentralCell is None:
            centralCells = self.centralCells
        else:
            if isinstance(manualCentralCell, int):
                manualCentralCell = np.asarray([manualCentralCell])
            elif isinstance(manualCentralCell, list):
                manualCentralCell = np.asarray(manualCentralCell)
            manualCentralCell = manualCentralCell.astype(str)
            isManualCellsInTable = np.isin(manualCentralCell, cellLabels)
            assert np.all(isManualCellsInTable), "Not all given cells are in the table. Cell/s {} are missing.".format(manualCentralCell[np.invert(isManualCellsInTable)])
            centralCells = manualCentralCell
        selectedRange = self.ensureAppropriateRange(centralCells, selectedRange)
        centralCells = centralCells[selectedRange]
        centralCells = centralCells.astype(str)
        if isinstance(selectedRange, int):
            isCentralCell = np.isin(cellLabels, centralCells)
            centralCellValue = isCentralCell.astype(int)
        else:
            centralCellValue = np.zeros(len(cellLabels))
            if sameValue:
                isCentralCell = np.isin(cellLabels, centralCells)
                centralCellValue[isCentralCell] = 1
            else:
                for i, centralCell in enumerate(centralCells[::-1]):
                    isCentralCell = np.isin(cellLabels, centralCell)
                    centralCellValue[isCentralCell] = i+1
        return centralCellValue

    def extractLastLines(self, fileToOpen, extractFooter=0):
        lastLines = ""
        file = open(fileToOpen, "r")
        allLines = file.readlines()
        file.close()
        if int(extractFooter) > 0:
            if len(allLines) >= extractFooter:
                lastLines = allLines[-extractFooter:]
                lastLines = "".join(lastLines)
                # do I need to change the range or the mesh number?
            else:
                print("The number of footer lines to extract ({}) was larger than the number of lines in the file {}.".format(extractFooter, fileToOpen))
        return lastLines

    def appendLinesToFile(self, lastLines, saveToFilename):
        file = open(str(saveToFilename), "a")
        file.write(lastLines)
        file.close()

def main():
    from pathlib import Path
    # filename = "Data/WT/P6/areaP6T4.csv"
    # table = pd.read_csv(filename)
    # table = table.iloc[:-2, :]
    # myCentralPositionFinder = CentralPositionFinder(table)
    # centralCellUsingMean = myCentralPositionFinder.GetCentralCell(measure="median", selectedRange=[0,1,2])
    # print(centralCellUsingMean)
    # sys.exit()
    # #vergleichen mit
    calculatedCentralCellsDictMean = {"P1":[], "P2":[], "P5":[], "P6":[], "P8":[]}
    calculatedCentralCellsDictMedian = {"P1":[], "P2":[], "P5":[], "P6":[], "P8":[]}
    centralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    dataFolder = "Data/WT/"
    skipFooter = 4
    saveManualCentralCells = False
    isNetworkTable = False
    if saveManualCentralCells:
        folder = "manualCentralPositions"# "centralPositions"
    elif isNetworkTable:
        folder = "centralPosNetwork"
    else:
        folder = "centralPositions"
    for key, items in centralCellsDict.items():
        Path(dataFolder + "{}/{}/".format(folder,key)).mkdir(parents=True, exist_ok=True)
        for i in range(len(items)):
            print("{}T{}.csv".format(key, i))
            filename = dataFolder + "{}/area{}T{}.csv".format(key, key, i)
            if isNetworkTable:
                table = dataFolder + "{}/cellularConnectivityNetwork{}T{}.csv".format(key, key, i)
            else:
                table = pd.read_csv(filename)#, skipfooter=skipFooter, engine="python")
                table = table.iloc[:-2, :]
            myCentralPositionFinder = CentralPositionFinder(table, isNetworkTable=isNetworkTable, showCoordDistribution=False)
            # centralCellUsingMean = myCentralPositionFinder.GetCentralCell(measure="mean", selectedRange=[0,1,2])
            # centralCellUsingMedian = myCentralPositionFinder.GetCentralCell(measure="median", selectedRange=[0,1,2])
            # calculatedCentralCellsDictMean[key].append(centralCellUsingMean)
            # calculatedCentralCellsDictMedian[key].append(centralCellUsingMedian)
            expectedCentralCell = centralCellsDict[key][i]
            # observedCentralCells = calculatedCentralCellsDictMean[key][i]
            # idxOfExpected = [np.where(table.iloc[:,0] == str(cellId))[0][0] for cellId in expectedCentralCell]
            # idxOfObserved = [np.where(table.iloc[:,0] == str(cellId))[0][0] for cellId in observedCentralCells]
            # expectedCentralPosition = np.mean(table.iloc[idxOfExpected, [3,4,5]], axis=0)
            # observedCentralPosition = table.iloc[idxOfObserved, [3,4,5]]
            # distances = myCentralPositionFinder.calcDistance(observedCentralPosition, expectedCentralPosition)
            # print(key, i, expectedCentralCell, observedCentralCells, distances)
            centralCellVisualisationFilename = dataFolder + "{}/{}/{}T{}.csv".format(folder, key, key, i)
            if saveManualCentralCells:
                selectedRange = np.arange(len(expectedCentralCell)) # np.arange(5)
                myCentralPositionFinder.SaveCentralCellAsHeatMap(centralCellVisualisationFilename,
                                                                 originalFilename=filename,
                                                                 manualCentralCell=np.asarray(expectedCentralCell), # None
                                                                 selectedRange=selectedRange)
            else:
                if isNetworkTable:
                    selectedRange = "max"
                    sameValue = True
                else:
                    selectedRange = np.arange(5)
                    sameValue = False
                myCentralPositionFinder.SaveCentralCellAsHeatMap(centralCellVisualisationFilename,
                                                                 originalFilename=filename,
                                                                 manualCentralCell=None,
                                                                 sameValue=sameValue,
                                                                 selectedRange=selectedRange)

if __name__ == '__main__':
    main()
