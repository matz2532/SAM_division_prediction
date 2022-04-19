import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BiologicalFeatureCreatorForNetworkRecreation (object):

    # create the biological feature set for topology prediction
    def __init__(self, baseFolder, networkRecreationFeatureTableFilename=None,
                 featureFilenameToSaveTo=None, usePlantNamesAsFolder=True,
                 plantNameIdx=0, timePointIdx=1, neighbourAndDividingCellIdx=[2, 3],
                 geometryTableBaseName="area{}T{}.csv",
                 sharedWallBaseName="cellularConnectivityNetwork{}T{}.csv",
                 skipFooterOfGeometryFile=4,
                 oldFeatureTable=None, centralCellsDict=None):
        self.baseFolder = baseFolder
        self.networkRecreationFeatureTableFilename = networkRecreationFeatureTableFilename
        self.featureFilenameToSaveTo = featureFilenameToSaveTo
        self.usePlantNamesAsFolder = usePlantNamesAsFolder
        self.plantNameIdx = plantNameIdx
        self.timePointIdx = timePointIdx
        self.neighbourAndDividingCellIdx = neighbourAndDividingCellIdx
        self.geometryTableBaseName = geometryTableBaseName
        self.sharedWallBaseName = sharedWallBaseName
        self.skipFooterOfGeometryFile = skipFooterOfGeometryFile
        self.centralCellsDict = centralCellsDict
        assert self.networkRecreationFeatureTableFilename is None or oldFeatureTable is None, "either networkRecreationFeatureTableFilename or oldFeatureTable need to be given to know which plants and time points to use."
        if not self.networkRecreationFeatureTableFilename is None:
            self.oldFeatureTable = pd.read_csv(self.networkRecreationFeatureTableFilename)
        else:
            self.oldFeatureTable = oldFeatureTable

    def CreateBiologicalFeatures(self):
        allCellFeatures = []
        plantNames = pd.unique(self.oldFeatureTable.iloc[:, self.plantNameIdx])
        timePoints = pd.unique(self.oldFeatureTable.iloc[:, self.timePointIdx])
        for self.plant in plantNames:
            for self.time in timePoints:
                if not self.centralCellsDict is None:
                    if self.plant in self.centralCellsDict:
                        if len(self.centralCellsDict[self.plant][self.time]) == 0:
                            continue
                cellFeatures = self.selectBiologicalFeaturesOfTissue(self.plant, self.time)
                if len(cellFeatures) > 0:
                    allCellFeatures.append(cellFeatures)
        cellFeatures = pd.concat(allCellFeatures, axis=0, ignore_index=True)
        cellIdentifierColIdndices = np.concatenate([[self.plantNameIdx], [self.timePointIdx], self.neighbourAndDividingCellIdx])
        biologicalFeatures = pd.concat([self.oldFeatureTable.iloc[:, cellIdentifierColIdndices], cellFeatures], axis=1)
        if not self.featureFilenameToSaveTo is None:
            biologicalFeatures.to_csv(self.featureFilenameToSaveTo, index=False)
        return biologicalFeatures

    def selectBiologicalFeaturesOfTissue(self, plant, time):
        isPlant = np.isin(self.oldFeatureTable.iloc[:, self.plantNameIdx], plant)
        isTime = np.isin(self.oldFeatureTable.iloc[:, self.timePointIdx], time)
        isSelectedTissue = isPlant & isTime
        idxOfSelectedTissueCells = np.where(isSelectedTissue)[0]
        cellFeatures = []
        geometryTable = self.loadFeatureTableOf(self.geometryTableBaseName, plant, time,
                                                skipFooter=self.skipFooterOfGeometryFile)
        sharedWallTable = self.loadFeatureTableOf(self.sharedWallBaseName, plant, time)
        # do I need to normalize shared wall?
        # plt.hist(sharedWallTable.iloc[:, 2])
        # plt.title("{} at {} with mean={}".format(plant, time, np.mean(sharedWallTable.iloc[:, 2])))
        # plt.show()
        # seems to look like poisson distribution, so I guess yes, but their means are very similar, so probably not necessary
        for cellIdx in idxOfSelectedTissueCells:
            dividingCell, neighbourCell = self.oldFeatureTable.iloc[cellIdx, self.neighbourAndDividingCellIdx]
            dividingCellArea = self.getAreaOf(dividingCell, geometryTable)
            neighbourCellArea = self.getAreaOf(neighbourCell, geometryTable)
            sharedWall = self.getSharedCellWallBetween(dividingCell, neighbourCell, sharedWallTable)
            distance = self.getDistanceBetween(dividingCell, neighbourCell, geometryTable)
            dividingCellPerimeter = self.getPerimeterOf(dividingCell, sharedWallTable)
            neighbourCellPerimeter = self.getPerimeterOf(neighbourCell, sharedWallTable)
            feature = [dividingCellArea, neighbourCellArea, sharedWall, distance, dividingCellPerimeter, neighbourCellPerimeter]
            cellFeatures.append(feature)
        cellFeatures = pd.DataFrame(cellFeatures, columns=["dividingCellArea", "neighbourCellArea", "sharedWall", "distance", "dividingCellPerimeter", "neighbourCellPerimeter"])
        return cellFeatures

    def loadFeatureTableOf(self, baseName, plant, time, skipFooter=0):
        assert len(baseName.split("{}")) >= 3, "There need to be at least two '{}' inside the baseName to ensure plant and time of the tissue can be properly inserted, {} < 3".format(len(baseName.split("{}")))
        filename = self.baseFolder + plant + "/" + baseName.format(plant, time)
        geometryTable = pd.read_csv(filename, skipfooter=skipFooter, engine="python")
        return geometryTable

    def getAreaOf(self, selectedCell, geometryTable, cellCol=0, areaCol=1):
        isCell = np.isin(geometryTable.iloc[:, cellCol], selectedCell)
        assert np.any(isCell), "The cell {} is not in the geometry table of {} time point {}. data type of selected cell is {}, needs to be {}.".format(selectedCell, self.plant, self.time, type(selectedCell), geometryTable.iloc[:, cellCol].dtypes)
        idxOfCell = np.where(isCell)[0]
        area = float(geometryTable.iloc[idxOfCell, areaCol])
        return area

    def getSharedCellWallBetween(self, cell1, cell2, sharedWallTable):
        isCell1 = np.isin(sharedWallTable.iloc[:, 0], cell1)
        isCell2 = np.isin(sharedWallTable.iloc[:, 1], cell2)
        assert np.any(isCell1), "The cell1 {} is not in the cellular connectivity table of {} time point {}".format(cell1, self.plant, self.time)
        assert np.any(isCell2), "The cell2 {} is not in the cellular connectivity table of {} time point {}".format(cell2, self.plant, self.time)
        areCellsAdjacent = isCell1 & isCell2
        assert np.any(areCellsAdjacent), "The cell {} and {} are not adjacent in the connectivity table of {} time point {}".format(cell1, cell2, self.plant, self.time)
        idxOfCell = np.where(areCellsAdjacent)[0]
        sharedWall = float(sharedWallTable.iloc[idxOfCell, 2])
        return sharedWall

    def getDistanceBetween(self, cell1, cell2, geometryTable, cellCol=0, xyzCoordIdx=[3, 4, 5]):
        isCell1 = np.isin(geometryTable.iloc[:, 0], cell1)
        isCell2 = np.isin(geometryTable.iloc[:, 0], cell2)
        assert np.any(isCell1), "The cell1 {} is not in the cellular connectivity table of {} time point {}".format(cell1, self.plant, self.time)
        assert np.any(isCell2), "The cell2 {} is not in the cellular connectivity table of {} time point {}".format(cell2, self.plant, self.time)
        idxOfCell1 = np.where(isCell1)[0]
        idxOfCell2 = np.where(isCell2)[0]
        coordOfCell1 = geometryTable.iloc[idxOfCell1, xyzCoordIdx].to_numpy()
        coordOfCell2 = geometryTable.iloc[idxOfCell2, xyzCoordIdx].to_numpy()
        distance = np.linalg.norm(coordOfCell1 - coordOfCell2)
        return distance

    def getPerimeterOf(self, selectedCell, sharedWallTable, cellCol=0, sharedWallCol=1):
        isCell = np.isin(sharedWallTable.iloc[:, cellCol], selectedCell)
        assert np.any(isCell), "The cell {} is not in the geometry table of {} time point {}".format(selectedCell, self.plant, self.time)
        neighbourIndices = np.where(isCell)[0]
        perimeter = 0
        for idxWithNeighbour in neighbourIndices:
            perimeter += sharedWallTable.iloc[idxWithNeighbour, sharedWallCol]
        return perimeter

def main():
    baseFolder = "./Data/WT/"
    filename = "Data/WT/topoPredData/manuallyCreated/defaultPar_concat_notnormalised.csv"
    featureFilenameToSaveTo = baseFolder + "topoPredData/biologicalFeatures/combinedBiologicalFeatures.csv"
    featureCreator = BiologicalFeatureCreatorForNetworkRecreation(baseFolder, filename,
                                featureFilenameToSaveTo)
    featureCreator.CreateBiologicalFeatures()

if __name__ == '__main__':
    main()
