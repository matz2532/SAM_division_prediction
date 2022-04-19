import copy
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "./Code/Predicting/")

from CentralPositionFinder import CentralPositionFinder
from FilenameCreator import FilenameCreator
from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from pathlib import Path
from StandardTableFormater import StandardTableFormater

class BaseDataCreator (object):

    def __init__(self, dataFolder=None, timePointsPerPlant=None, plantNames=None,
                folderToSave=None, sep=",", skipFooterOfGeometryFile=4,
                specialGraphProperties=None, centralCellsDict=None, centerRadius=30,
                zNormaliseFeaturesPerTissue=False, onlyAra=False, printProgress=True):
        self.dataFolder = dataFolder
        self.timePointsPerPlant = timePointsPerPlant
        self.plantNames = plantNames
        self.folderToSave = folderToSave
        self.sep = sep
        self.specialGraphProperties = specialGraphProperties
        self.centralCellsDict = centralCellsDict
        self.centerRadius = centerRadius
        self.myStandardTableFormater = StandardTableFormater(self.plantNames)
        self.skipFooterOfGeometryFile = skipFooterOfGeometryFile
        self.onlyAra = onlyAra
        self.printProgress = printProgress
        self.zNormaliseFeaturesPerTissue = zNormaliseFeaturesPerTissue = None
        self.labelTable, self.featureTable = None, None
        if not dataFolder is None and not timePointsPerPlant is None and not plantNames is None:
            self.nrOfPlantNames = len(self.plantNames)
            self.totalNrOfSamples = self.nrOfPlantNames * self.timePointsPerPlant
            self.setFilenames(self.dataFolder, self.timePointsPerPlant, self.plantNames)
            self.setupData(centralCellsDict)

    def setFilenames(self, dataFolder, timePointsPerPlant, plantNames,
                     lengthOfTimeStep=1, usePlantNamesAsFolder=True):
        myFilenameCreator = FilenameCreator(dataFolder, timePointsPerPlant,
                    plantNames, usePlantNamesAsFolder=usePlantNamesAsFolder,
                    addlastTimePoint=True, lengthOfTimeStep=lengthOfTimeStep,
                    connectivityText="cellularConnectivityNetwork",
                    parentLabelingText="parentLabeling", geometryText="area",
                    peripheralLabelText="periphery labels ")
        self.connectivityNetworkFilenames = np.asarray(myFilenameCreator.GetConnectivityNetworkFilenames())
        self.areaFilenames = np.asarray(myFilenameCreator.GetAreaFilenames())
        self.peripheralLabelsFilename = myFilenameCreator.GetPeripheralLabelsFilenames()
        self.parentLabelingFilenames = np.asarray(myFilenameCreator.GetPartentLabellingFilenames())

    def setupData(self, centralCellsDict=None):
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
            plantData["parentDaugherCellLabeling"] = self.parentLabelingFilenames[filenameIdxOfPlant]
            self.data[plantName] = plantData
            if not centralCellsDict is None:
                plantData["centralCells"] = centralCellsDict[plantName]
                self.checkExistenceOfCentralCellsInNetwork(centralCellsDict[plantName],
                        graphCreators,
                        plantName)
            else:
                plantData["centralCells"] = self.calcCentralCellsFor(plantData["geometryFilename"])
            self.data[plantName] = plantData

    def testForCorrectNumberOfFilenames(self):
        expectedNumberOfFilenames = self.timePointsPerPlant * len(self.plantNames)
        assert expectedNumberOfFilenames == len(self.connectivityNetworkFilenames), "There are cellular connectivity networkx filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.connectivityNetworkFilenames))
        assert expectedNumberOfFilenames == len(self.areaFilenames), "There are area filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.areaFilenames))
        assert expectedNumberOfFilenames == len(self.peripheralLabelsFilename), "There are peripheral label filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.peripheralLabelsFilename))
        assert expectedNumberOfFilenames == len(self.parentLabelingFilenames), "There are parent labeling filenames missing in self.connectivityNetworkFilenames, {} != {}".format(expectedNumberOfFilenames, len(self.parentLabelingFilenames))

    def calcCellNetworksOf(self, filenameIdxOfPlant):
        graphCreators = []
        for filenameIdx in filenameIdxOfPlant:
            filenameCellNetwork = self.connectivityNetworkFilenames[filenameIdx]
            if Path(filenameCellNetwork).is_file():
                graphCreator = GraphCreatorFromAdjacencyList(filenameCellNetwork,
                                            skipFooter=self.skipFooterOfGeometryFile)
                geometryFilename = self.areaFilenames[filenameIdx]
                graphCreator.AddCoordinatesPropertyToGraphFrom(geometryFilename)
            else:
                graphCreator = None
            graphCreators.append(graphCreator)
        return graphCreators

    def checkExistenceOfCentralCellsInNetwork(self, nestedCentralNodesList, networks, plantName):
        for i, nodesList in enumerate(nestedCentralNodesList):
            if not networks[i] is None:
                networkNodes = list(networks[i].GetGraph().nodes())
                isNodeInNetwork = np.isin(nodesList, networkNodes)
                assert np.all(isNodeInNetwork), "In plant {} time {} the nodes {} of {} are not present in the network from '{}' with nodes named: {}".format(plantName, i, np.asarray(nodesList)[np.invert(isNodeInNetwork)], nodesList, self.data[plantName]["graphFilename"][i], np.sort(networkNodes))
            else:
                assert len(nodesList) == 0, f"The f{i}th network of {plantName} does not exist, but it got the central cells {nodesList} given."

    def calcCentralCellsFor(self, geometryTableFilenames):
        centralCellsPerTimePoint = []
        for filename in geometryTableFilenames:
            table = pd.read_csv(filename, sep=self.sep, engine="python",
                                skipfooter=self.skipFooterOfGeometryFile)
            centrumFinder = CentralPositionFinder(table)
            centralCell = centrumFinder.GetCentralCell()
            centralCellsPerTimePoint.append(centralCell)
        return centralCellsPerTimePoint

    def MakeTrainingData(self, folderToSave=None, estimateLabels=True,
                         estimateFeatures=True, skipEmptyCentrals=False):
        raise NotImplementedError("The inheriting class should implement this functionality.")
