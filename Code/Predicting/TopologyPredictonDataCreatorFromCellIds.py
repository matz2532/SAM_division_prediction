import copy
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./Code/Feature and Label Creation/")
sys.path.insert(0, "./Code/")

from BaseDataCreator import BaseDataCreator
from BiologicalFeatureCreatorForNetworkRecreation import BiologicalFeatureCreatorForNetworkRecreation
from CentralPositionFinder import CentralPositionFinder
from FeatureVectorCreator import FeatureVectorCreator
from FilenameCreator import FilenameCreator
from utils import FooterExtractor
from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from NeighborsOfDividingCellMapper import NeighborsOfDividingCellMapper
from utils import convertParentLabelingTableToDict
from pathlib import Path
from StandardTableFormater import StandardTableFormater
from TopologyPredictonDataCreator import TopologyPredictonDataCreator

class TopologyPredictonDataCreatorFromCellIds (TopologyPredictonDataCreator):

    useRatio=False
    useAbsDifferenceInFeatures=False
    zNormaliseFeaturesPerTissue=False

    def __init__(self, selectedCellsOfTissue, dataFolder=None, timePointsPerPlant=None, plantNames=None,
                folderToSave=None, sep=",", skipFooterOfGeometryFile=4,
                specialGraphProperties=None, useDifferenceInFeatures=True,
                concatParentFeatures=True):
        self.selectedCellsOfTissue = selectedCellsOfTissue
        self.dataFolder = dataFolder
        self.timePointsPerPlant = timePointsPerPlant
        self.plantNames = plantNames
        self.sep = sep
        self.specialGraphProperties = specialGraphProperties
        self.myStandardTableFormater = StandardTableFormater(self.plantNames)
        self.skipFooterOfGeometryFile = skipFooterOfGeometryFile
        self.useDifferenceInFeatures = useDifferenceInFeatures
        self.concatParentFeatures = concatParentFeatures
        if not dataFolder is None and not timePointsPerPlant is None and not plantNames is None:
            self.nrOfPlantNames = len(self.plantNames)
            self.totalNrOfSamples = self.nrOfPlantNames * self.timePointsPerPlant
            self.setFilenames(self.dataFolder, self.timePointsPerPlant, self.plantNames)
            self.setupData()

    def setupData(self):
        self.testForCorrectNumberOfFilenames()
        self.data = {}
        for plantIdx, plantName in enumerate(self.plantNames):
            plantData = {}
            filenameIdxOfPlant = np.arange(start=plantIdx, stop=self.totalNrOfSamples,
                                           step=len(self.plantNames))
            graphCreators = self.calcCellNetworksOf(filenameIdxOfPlant)
            plantData["graphCreators"] = graphCreators
            plantData["geometryFilename"] = self.areaFilenames[filenameIdxOfPlant]
            plantData["graphFilename"] = self.connectivityNetworkFilenames[filenameIdxOfPlant]
            self.data[plantName] = plantData

    def testForCorrectNumberOfFilenames(self):
        expectedNumberOfFilenames = self.timePointsPerPlant * len(self.plantNames)
        assert expectedNumberOfFilenames == len(self.connectivityNetworkFilenames), "There are cellular connectivity networkx filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.connectivityNetworkFilenames))
        assert expectedNumberOfFilenames == len(self.areaFilenames), "There are area filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.areaFilenames))
        assert expectedNumberOfFilenames == len(self.peripheralLabelsFilename), "There are peripheral label filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.peripheralLabelsFilename))

    def CalcFeaturesForGivenCells(self):
        self.dataOfPlantTissue = self.calcDividingNeighbourCellPairsTable()
        self.featureTable = self.calcFeatures()

    def calcDividingNeighbourCellPairsTable(self):
        dataOfPlantTissue = []
        for plantIdx, plantName in enumerate(self.plantNames):
            plantName = self.plantNames[plantIdx]
            for timeIdx in range(self.timePointsPerPlant-1):
                parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx)
                selectedCells = self.selectedCellsOfTissue[plantName][timeIdx]
                selectedCellsNeighbourPairs = self.getCellNeighbourPairs(selectedCells, parentConnectivityNetwork)
                self.myStandardTableFormater.SetPlantNameByIdx(plantIdx)
                self.myStandardTableFormater.SetTimePoint(timeIdx)
                for dividingParentCell, neighbour in selectedCellsNeighbourPairs:
                    mappedDaughterData = [dividingParentCell, neighbour, neighbour]
                    out = self.myStandardTableFormater.GetProperStandardFormatWith(mappedDaughterData)
                    dataOfPlantTissue.append(out)
        columnNames = ["plant", "time point", "dividing parent cell", "parent neighbor", "mapped daughter"]
        dataOfPlantTissue = pd.DataFrame(dataOfPlantTissue, columns=columnNames)
        return dataOfPlantTissue

    def getCellNeighbourPairs(self, cellIds, network):
        selectedCellsNeighbourPairs = []
        for cell in cellIds:
            neighbours = list(network.neighbors(cell))
            for neighbourCell in neighbours:
                selectedCellsNeighbourPairs.append([cell, neighbourCell])
        selectedCellsNeighbourPairs = np.asarray(selectedCellsNeighbourPairs)
        return selectedCellsNeighbourPairs

    def calcFeatures(self):
        groups = self.dataOfPlantTissue.groupby(["plant", "time point"])
        allFeatures = []
        for tissueId, tissueOverview in groups:
            featureOfTissue = self.determineFeaturesOf(tissueOverview, tissueId[0], tissueId[1])
            allFeatures.append(featureOfTissue)
        allFeatures = pd.concat(allFeatures, axis=0, ignore_index=True)
        return allFeatures

    def calcFeatureTable(self, plantIdx, timeIdx, uniqueMappedParentCells):
        if self.specialGraphProperties is None:
            parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx, isPlantIdxPlantName=True)
        else:
            parentConnectivityNetwork = self.calcSpecialGraph(plantIdx, timeIdx, isPlantIdxPlantName=True)
        myFeatureVectorCreator = FeatureVectorCreator(parentConnectivityNetwork,
                                                      allowedLabels=uniqueMappedParentCells,
                                                      zNormaliseFeaturesPerTissue=self.zNormaliseFeaturesPerTissue)
        featureDF = myFeatureVectorCreator.GetFeatureMatrixDataFrame()
        featureDF.index = np.arange(len(uniqueMappedParentCells))
        return featureDF

    def SaveFeatureTable(self, filenameToSave, sep=","):
        Path(filenameToSave).parent.mkdir(parents=True, exist_ok=True)
        self.featureTable.to_csv(filenameToSave, index=False, sep=sep)

def saveTopoFeatureSetsFromCellIds(set, dataFolder, plantNames, folderToSave, selectedCellsOfTissue,
                timePointsPerPlant=5):
    from CreateFeatureSets import CreateFeatureSets
    specialGraphProperties = None
    featureProperty = "topology"
    if set == 1:
        # for area
        featureProperty = "topologyArea"
        specialGraphProperties = {"useEdgeWeight":True,
                                  "invertEdgeWeight": True,
                                  "useSharedWallWeight": False,
                                  "useDistanceWeight": False,
                                  "maxNormEdgeWeightPerGraph": False}
    elif set == 2:
        # for Wall
        featureProperty = "topologyWall"
        specialGraphProperties = {"useEdgeWeight":True,
                                  "invertEdgeWeight": False,
                                  "useSharedWallWeight": True,
                                  "useDistanceWeight": False,
                                  "maxNormEdgeWeightPerGraph": False}
    elif set == 3:
        # for distance
        featureProperty = "topologyDist"
        specialGraphProperties = {"useEdgeWeight":True,
                                  "invertEdgeWeight": False,
                                  "useSharedWallWeight": False,
                                  "useDistanceWeight": True,
                                  "maxNormEdgeWeightPerGraph": False}
    elif set == 4:
        featureProperty = "bio"
        Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
        otherFeatureFilename = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format("topology", "topology")
        otherFeatureTable = pd.read_csv(otherFeatureFilename)
        filenameToSave = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featureProperty, featureProperty)
        BiologicalFeatureCreatorForNetworkRecreation(baseFolder=dataFolder,
                featureFilenameToSaveTo=filenameToSave,
                oldFeatureTable=otherFeatureTable).CreateBiologicalFeatures()
        return None
    elif set == 5:
        featureProperty = "allTopos"
        Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
        CreateFeatureSets(dataFolder=None, folderToSave=None, setRange=[]).combineTopoFeatures(folderToSave, featureProperty)
        return None
    elif set == 6:
        featureProperty = "topoAndBio"
        Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
        CreateFeatureSets(dataFolder=None, folderToSave=None, setRange=[]).combineTopoFeatures(folderToSave, featureProperty, includeBioFeatures=True)
        return None
    myTopologyPredictonDataCreator = TopologyPredictonDataCreatorFromCellIds(selectedCellsOfTissue,
                                            dataFolder, timePointsPerPlant,
                                            plantNames,
                                            specialGraphProperties=specialGraphProperties)
    myTopologyPredictonDataCreator.CalcFeaturesForGivenCells()
    folderToSave += featureProperty + "/"
    Path(folderToSave).mkdir(parents=True, exist_ok=True)
    myTopologyPredictonDataCreator.SaveFeatureTable(folderToSave+"combinedFeatures_{}_notnormalised.csv".format(featureProperty))

def main():
    import pickle
    setIdx = 0
    plant = "P2"
    savePredictionsToFolder = f"Temporary/DivAndTopoApplication/{plant}/"
    loadPredictionsToFolder = f"Results/DivAndTopoApplication/{plant}/"
    dividingCellsOfTimePoint = pickle.load(open(loadPredictionsToFolder+"dividingCellsOfTimePoint.pkl", "rb"))
    selectedCellsOfTissue = {plant : dividingCellsOfTimePoint}
    saveTopoFeatureSetsFromCellIds(setIdx, "Data/WT/", [plant], savePredictionsToFolder,
                                   selectedCellsOfTissue, timePointsPerPlant=5)

if __name__== "__main__":
    main()
