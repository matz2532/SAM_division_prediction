import copy
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./Code/Feature and Label Creation/")
sys.path.insert(0, "./Code/")

from BaseDataCreator import BaseDataCreator
from CentralPositionFinder import CentralPositionFinder
from FeatureVectorCreator import FeatureVectorCreator
from FilenameCreator import FilenameCreator
from utils import FooterExtractor
from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from NeighborsOfDividingCellMapper import NeighborsOfDividingCellMapper
from utils import convertParentLabelingTableToDict
from pathlib import Path
from StandardTableFormater import StandardTableFormater

class TopologyPredictonDataCreator (BaseDataCreator):

    def __init__(self, dataFolder=None, timePointsPerPlant=None, plantNames=None,
                folderToSave=None, sep=",", skipFooterOfGeometryFile=4,
                specialGraphProperties=None, centralCellsDict=None,
                useRatio=False, useDifferenceInFeatures=False, useAbsDifferenceInFeatures=False,
                concatParentFeatures=True,
                zNormaliseFeaturesPerTissue=False):
        self.dataFolder = dataFolder
        self.timePointsPerPlant = timePointsPerPlant
        self.plantNames = plantNames
        self.sep = sep
        self.specialGraphProperties = specialGraphProperties
        self.centralCellsDict = centralCellsDict
        self.myStandardTableFormater = StandardTableFormater(self.plantNames)
        self.skipFooterOfGeometryFile = skipFooterOfGeometryFile
        self.useRatio = useRatio
        self.useDifferenceInFeatures = useDifferenceInFeatures
        self.useAbsDifferenceInFeatures = useAbsDifferenceInFeatures
        self.concatParentFeatures = concatParentFeatures
        self.zNormaliseFeaturesPerTissue = zNormaliseFeaturesPerTissue
        self.labelTable, self.featureTable = None, None
        if not dataFolder is None and not timePointsPerPlant is None and not plantNames is None:
            self.nrOfPlantNames = len(self.plantNames)
            self.totalNrOfSamples = self.nrOfPlantNames * self.timePointsPerPlant
            self.setFilenames(self.dataFolder, self.timePointsPerPlant, self.plantNames)
            self.setFullParentLabelingFilenames(self.dataFolder, self.timePointsPerPlant, self.plantNames)
            self.setupData(centralCellsDict)
            self.addParentLabelingData(centralCellsDict)

    def setFullParentLabelingFilenames(self, dataFolder, timePointsPerPlant, plantNames,
                 lengthOfTimeStep=1, usePlantNamesAsFolder=True):
        myFilenameCreator = FilenameCreator(dataFolder, timePointsPerPlant,
            plantNames, usePlantNamesAsFolder=usePlantNamesAsFolder,
            addlastTimePoint=True, lengthOfTimeStep=lengthOfTimeStep,
            parentLabelingText="fullParentLabeling")
        self.fullParentLabelingFilenames = myFilenameCreator.GetPartentLabellingFilenames()

    def addParentLabelingData(self, centralCellsDict=None):
        for plantIdx, plantName in enumerate(self.plantNames):
            plantData = self.data[plantName]
            filenameIdxOfPlant = np.arange(start=plantIdx, stop=self.totalNrOfSamples,
                                           step=len(self.plantNames))
            graphCreators = plantData["graphCreators"]
            plantData["parentDaugherCellLabelingFilenames"] = plantData["parentDaugherCellLabeling"].copy()
            plantData["fullParentDaugherCellLabelingFilenames"] = np.asarray(self.fullParentLabelingFilenames)[filenameIdxOfPlant]
            parentDaugherCellLabeling = self.calcParentDaughterMappingOfDividingCells(filenameIdxOfPlant[:-1], graphCreators)
            plantData["parentDaugherCellLabeling"] = parentDaugherCellLabeling
            plantData["fullParentLabeling"] = self.calcParentDaughterMappingOfDividingCells(filenameIdxOfPlant[:-1],
                                                        graphCreators,
                                                        useFullParentLabeling=True)

    def calcParentDaughterMappingOfDividingCells(self, filenameIdxOfPlant, graphCreators, useFullParentLabeling=False):
        parentDaugherCellLabeling = []
        for filenameIdx in filenameIdxOfPlant:
            parentDaughterDict = {}
            if useFullParentLabeling:
                filename = self.fullParentLabelingFilenames[filenameIdx]
            else:
                filename = self.parentLabelingFilenames[filenameIdx]
            if Path(filename).is_file():
                parentLabelingTable = pd.read_csv(filename, sep=self.sep)
                parentDaughterDict = convertParentLabelingTableToDict(parentLabelingTable)
            parentDaugherCellLabeling.append(parentDaughterDict)
        return parentDaugherCellLabeling

    def MakeTrainingData(self, folderToSave=None, estimateLabels=True,
                         estimateFeatures=True, skipEmptyCentrals=False):
        if estimateLabels or estimateFeatures:
            self.addMappedCellsToData()
        self.estimateLabels = estimateLabels
        self.estimateFeatures = estimateFeatures
        if self.estimateLabels is True or self.estimateLabels is None:
            self.calcLabels()
        elif not self.estimateLabels is None:
            self.labelTable = pd.read_csv(self.estimateLabels)
        if self.estimateFeatures:
            self.calcFeaturesOfMappedParentCells()

    def addMappedCellsToData(self):
        for plantIdx, plantName in enumerate(self.plantNames):
            plantName = self.plantNames[plantIdx]
            self.data[plantName]["mappedCells"] = []
            for timeIdx in range(self.timePointsPerPlant-1):
                mappedCells = []
                fullParentLabeling = self.GetValuesFrom("fullParentLabeling", plantIdx, timeIdx)
                parentDaughterLabeling = self.GetValuesFrom("parentDaugherCellLabeling", plantIdx, timeIdx)
                centralCellIsGiven = len(self.centralCellsDict[plantName][timeIdx]) > 0
                if bool(fullParentLabeling) and bool(parentDaughterLabeling) and centralCellIsGiven: # if both dicts are not empty and central cell is given continue
                    parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx)
                    daughterConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx+1)
                    myMapper = NeighborsOfDividingCellMapper(parentConnectivityNetwork,
                                    daughterConnectivityNetwork,
                                    parentDaughterLabeling, fullParentLabeling,
                                    plant=plantName, timePoint=timeIdx)
                    mappedCells = myMapper.GetMappedCells()
                else:
                    graphCreators = self.data[plantName]["graphCreators"]
                    centralCells = self.centralCellsDict[plantName][timeIdx]
                    if not graphCreators[timeIdx] is None and graphCreators[timeIdx+1] is None and len(centralCells) > 0:
                        print(f"""WARNING: The parent labeling files should have been present for the time step of {plantName} T{timeIdx}T{timeIdx+1} with the given central cells {centralCells} and therefore was ignored.
                                          The expected filenames are '{self.data[plantName]["fullParentDaugherCellLabelingFilenames"][timeIdx]}' and '{self.data[plantName]["parentDaugherCellLabelingFilenames"][timeIdx]}'""")
                self.data[plantName]["mappedCells"].append(mappedCells)

    def GetValuesFrom(self, value, plantIdx, timeIdx, isPlantIdxPlantName=False):
        validValues = ["graph", "graphCreators", "parentDaugherCellLabeling",
                       "mappedCells", "geometryFilename", "graphFilename",
                       "centralCells", "fullParentLabeling"]
        assert value in validValues, "The value {} you choose needs to be one of the following values {}".format(value, validValues)
        if isPlantIdxPlantName:
            assert plantIdx in self.data, "plantIdx needs to be a plant name (key) in self.data if isPlantIdxPlantName is True. {} is not in {}".format(plantIdx, self.data.keys())
            plantName = plantIdx
        else:
            assert type(plantIdx) == int or type(plantIdx) == np.int64, "plantIdx needs to be an integer. {} is of type {} != int".format(plantIdx, type(plantIdx))
            plantName = self.plantNames[plantIdx]
        plantData = self.data[plantName]
        assert value != "mappedCells" or value in plantData, "mappedCells is not yet calculated."
        if value == "graph":
            assert timeIdx < len(plantData["graphCreators"]), "Time index is out of range. {} < {}".format(timeIdx, len(plantData[value][timeIdx]))
            return plantData["graphCreators"][timeIdx].GetGraph()
        else:
            assert timeIdx < len(plantData[value]), "Time index is out of range. {} < {}".format(timeIdx, len(plantData[value][timeIdx]))
            return plantData[value][timeIdx]

    def calcLabels(self):
        neighborLabelAndSource = []
        for plantIdx in range(len(self.plantNames)):
            for timeIdx in range(self.timePointsPerPlant-1):
                labelsOfTissue = self.determineLabelsOf(plantIdx, timeIdx)
                if len(labelsOfTissue) > 0:
                    neighborLabelAndSource.extend(labelsOfTissue)
        columnNames = ["plant", "time point", "dividing parent cell", "parent neighbor", "mapped daughter", "label"]
        self.labelTable = pd.DataFrame(neighborLabelAndSource, columns=columnNames)

    def determineLabelsOf(self, plantIdx, timeIdx, labelType="center distance",
                          printDetailsToNotConnected=True):
        self.myStandardTableFormater.SetPlantNameByIdx(plantIdx)
        self.myStandardTableFormater.SetTimePoint(timeIdx)
        mappedCells = self.GetValuesFrom("mappedCells", plantIdx, timeIdx)
        if len(mappedCells) == 0:
            return []
        self.parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx)# do I really need this?
        self.daughterConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx+1)
        self.parentDaugherCellLabeling = self.GetValuesFrom("fullParentLabeling", plantIdx, timeIdx)
        if labelType == "center distance":
            centralCells = self.GetValuesFrom("centralCells", plantIdx, timeIdx)
            centralCoordinate = self.calcCentralPosition(centralCells)
        else:
            centralCoordinate = None
        labelDataOfPlantTissue = []
        notConnectedCorrectly = 0
        for dividingParentCell, mappedNeighbors in mappedCells.items():
            dividedDaughterCells = self.parentDaugherCellLabeling[dividingParentCell]
            if len(dividedDaughterCells) == 2:
                cellOne, cellTwo = self.daughterCellLabeling(dividingParentCell, centralCoordinate)
                for mappedParent, mappedDaughter in mappedNeighbors.items():
                    daughterNeighbors = self.calcDaughterNeigborsOfCurrentNetworkFor(mappedParent, mappedDaughter, np.asarray([cellOne, cellTwo]))
                    if self.isAtLeastOneIn(dividedDaughterCells, daughterNeighbors):
                        label = self.determineLabelBasedOnConnectionFor(mappedDaughter,
                                                    daughterNeighbors, cellOne, cellTwo)
                        mappedDaughterData = [dividingParentCell, mappedParent, mappedDaughter, label]
                        out = self.myStandardTableFormater.GetProperStandardFormatWith(mappedDaughterData)
                        labelDataOfPlantTissue.append(out)
                    else:
                        notConnectedCorrectly += 1
                        if printDetailsToNotConnected:
                            print("mappedDaughter", mappedDaughter, " has neighbors", daughterNeighbors, "and should be adjacent to divided cells", cellOne, cellTwo, "which divided from", dividingParentCell, "; {} divids to {}".format(mappedParent, self.parentDaugherCellLabeling[mappedParent]) if mappedParent in self.parentDaugherCellLabeling else "")
                            print("No connection found for mapped parent {} on daughter {} in plant {} time point {}".format(mappedParent, mappedDaughter, self.plantNames[plantIdx], timeIdx))
                            self.saveBadAdjacency(plantIdx, timeIdx, dividedDaughterCells, mappedDaughter)
        if notConnectedCorrectly > 0:
            print("In {} {} were {} pairs excluded due to not being connected correctly".format(self.plantNames[plantIdx], timeIdx, notConnectedCorrectly))
        return labelDataOfPlantTissue

    def saveBadAdjacency(self, plantIdx, timeIdx, dividedDaughterCells, mappedDaughter, skipfooter=4):
        # mark cells in heat map to display divided daughter cells and "the not anymore connected" daughter cell
        positionalFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx+1)
        positionalTable = pd.read_csv(positionalFilename, engine="python", skipfooter=skipfooter)
        labels = positionalTable.iloc[:, 0].to_numpy()
        idxDivDaughter = np.where(np.isin(labels, dividedDaughterCells))[0]
        idxNotConnectedDaughterCell = np.where(np.isin(labels, mappedDaughter))[0]
        positionalTable.iloc[:, 1] = 0
        positionalTable.iloc[idxDivDaughter, 1] = 2
        positionalTable.iloc[idxNotConnectedDaughterCell, 1] = 1
        folderToSave = Path(positionalFilename).parent
        filenameToSave = "{}/checkDaughterAdjacency{}T{}Of{}With{}.csv".format(folderToSave, self.plantNames[plantIdx], timeIdx+1, mappedDaughter, dividedDaughterCells)
        positionalTable.to_csv(filenameToSave, index=False)
        FooterExtractor(fileToOpen=positionalFilename, extractFooter=skipfooter,  saveToFilename=filenameToSave)
        print("saved", filenameToSave)

    def calcCentralPosition(self, centralCells):
        cellPositions = self.parentConnectivityNetwork.nodes(data=True)
        centralCoord = []
        for cell in centralCells:
            coordinate = list(cellPositions[cell].values())
            centralCoord.append(coordinate)
        return np.mean(centralCoord, axis=0)

    def daughterCellLabeling(self, dividingParentCell, centralCoordinate=None):
        parentCellPosition = self.getNetworkDataValuesOfNode(self.parentConnectivityNetwork, dividingParentCell)
        dividingDaughterCells = self.parentDaugherCellLabeling[dividingParentCell]
        daughterCellPositionOne = self.getNetworkDataValuesOfNode(self.daughterConnectivityNetwork, dividingDaughterCells[0])
        daughterCellPositionTwo = self.getNetworkDataValuesOfNode(self.daughterConnectivityNetwork, dividingDaughterCells[1])
        if centralCoordinate is None:
            distanceOfCellOne = np.linalg.norm(parentCellPosition-daughterCellPositionOne)
            distanceOfCellTwo = np.linalg.norm(parentCellPosition-daughterCellPositionTwo)
        else:
            distanceOfCellOne = np.linalg.norm(centralCoordinate-daughterCellPositionOne)
            distanceOfCellTwo = np.linalg.norm(centralCoordinate-daughterCellPositionTwo)
        if distanceOfCellOne < distanceOfCellTwo:
            return dividingDaughterCells[0], dividingDaughterCells[1]
        else:
            return dividingDaughterCells[1], dividingDaughterCells[0]

    def getNetworkDataValuesOfNode(self, network, node):
        return np.asarray(list(network.nodes(data=True)[node].values()))

    def calcDaughterNeigborsOfCurrentNetworkFor(self, mappedParent, mappedDaughter, dividedNeighbours):
        isMappedParentDividing = mappedParent in self.parentDaugherCellLabeling
        if isMappedParentDividing:
            correctDaughterCells = self.parentDaugherCellLabeling[mappedParent]
            daughterNeighbors = np.concatenate([list(self.daughterConnectivityNetwork.neighbors(c)) for c in correctDaughterCells])
        else:
            daughterNeighbors = list(self.daughterConnectivityNetwork.neighbors(mappedDaughter))
        return daughterNeighbors

    def isAtLeastOneIn(self, dividedDaughterCells, daughterNeighbors):
        return np.sum(np.isin(dividedDaughterCells, daughterNeighbors)) > 0

    def determineLabelBasedOnConnectionFor(self, node, neighbors, cellOne, cellTwo):
        isCellOneNeighbor = cellOne in neighbors
        isCellTwoNeighbor = cellTwo in neighbors
        if isCellOneNeighbor and isCellTwoNeighbor:
            return  2
        elif isCellOneNeighbor:
            return 0
        elif isCellTwoNeighbor:
            return 1

    def calcFeaturesOfMappedParentCells(self):
        self.myStandardTableFormater = StandardTableFormater(self.plantNames)
        columnNames = ["plant", "time point", "parent neighbor"]
        featuresAndSources = []
        plantNames = self.labelTable.iloc[:, 0].unique()
        timePoints = self.labelTable.iloc[:, 1].unique()
        for plantName in plantNames:
            for timePoint in timePoints:
                tissueLabelTable = self.getTissueLabels(plantName, timePoint)
                plantIdx = np.where(np.asarray(self.plantNames) == plantName)[0][0]
                timeIdx = np.where(timePoints == timePoint)[0][0]
                if len(tissueLabelTable) > 0:
                    featuresOfTissue = self.determineFeaturesOf(tissueLabelTable, plantIdx, timeIdx)
                    featuresAndSources.append(featuresOfTissue)
        self.featureTable = pd.concat(featuresAndSources, axis=0, ignore_index=True)

    def getTissueLabels(self, plantName, timePoint, plantColIdx=0, timeColIdx=1):
        isPlant = np.isin(self.labelTable.iloc[:, plantColIdx], plantName)
        isTime = np.isin(self.labelTable.iloc[:, timeColIdx], timePoint)
        isTissue = isPlant & isTime
        if np.any(isTissue):
            idxOfCurrentTissue = np.where(isTissue)[0]
            tissueLabelTable = self.labelTable.iloc[idxOfCurrentTissue, :]
        else:
            tissueLabelTable = []
        return tissueLabelTable

    def determineFeaturesOf(self, tissueLabelTable, plantIdx, timeIdx):
        fullFormat = tissueLabelTable.to_numpy()
        dividingParentCells = fullFormat[:, 2].astype(int)
        neighboringParentCells = fullFormat[:, 3].astype(int)
        unique = np.unique([dividingParentCells, neighboringParentCells])
        uniqueFeatureDF = self.calcFeatureTable(plantIdx, timeIdx, unique)
        isNeighbore = self.whereAreElementsIn(unique, neighboringParentCells)
        isCentral = self.whereAreElementsIn(unique, dividingParentCells)
        featuresOfNeighbors = uniqueFeatureDF.iloc[isNeighbore,:].copy()
        featuresOfCentralCells = uniqueFeatureDF.iloc[isCentral,:].copy()
        featuresOfNeighbors.index = np.arange(len(featuresOfNeighbors))
        featuresOfCentralCells.index = np.arange(len(featuresOfCentralCells))
        combinedFeatures = [tissueLabelTable.iloc[:, :4].copy()]
        combinedFeatures[0].index = np.arange(len(combinedFeatures[0]))
        if self.useRatio:
            combinedFeatures.append(featuresOfNeighbors/featuresOfCentralCells)
            #add diff and abs diff for automatic calculation (for now the calculations are done manually see test.py "# create different versions of features")
        elif self.useDifferenceInFeatures:
            combinedFeatures.append(featuresOfNeighbors-featuresOfCentralCells)
        elif self.useAbsDifferenceInFeatures:
            combinedFeatures.append(np.abs(featuresOfNeighbors-featuresOfCentralCells))
        else:
            combinedFeatures.append(featuresOfNeighbors)
        if self.concatParentFeatures:
            combinedFeatures.append(featuresOfCentralCells)
        combinedFeatureDF = pd.concat(combinedFeatures, axis=1)
        return combinedFeatureDF

    def calcFeatureTable(self, plantIdx, timeIdx, uniqueMappedParentCells):
        if self.specialGraphProperties is None:
            parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx)
        else:
            parentConnectivityNetwork = self.calcSpecialGraph(plantIdx, timeIdx)
            cellSizeFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx)
        myFeatureVectorCreator = FeatureVectorCreator(parentConnectivityNetwork,
                                                      allowedLabels=uniqueMappedParentCells,
                                                      zNormaliseFeaturesPerTissue=self.zNormaliseFeaturesPerTissue)
        featureDF = myFeatureVectorCreator.GetFeatureMatrixDataFrame()
        featureDF.index = np.arange(len(uniqueMappedParentCells))
        return featureDF

    def calcSpecialGraph(self, plantIdx, timeIdx, isPlantIdxPlantName=False):
        cellSizeFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx, isPlantIdxPlantName=isPlantIdxPlantName)
        graphFilename = self.GetValuesFrom("graphFilename", plantIdx, timeIdx, isPlantIdxPlantName=isPlantIdxPlantName)
        useEdgeWeight = self.specialGraphProperties["useEdgeWeight"]
        invertEdgeWeight = self.specialGraphProperties["invertEdgeWeight"]
        useSharedWallWeight = self.specialGraphProperties["useSharedWallWeight"]
        maxNormEdgeWeightPerGraph = self.specialGraphProperties["maxNormEdgeWeightPerGraph"]
        useDistanceWeight = self.specialGraphProperties["useDistanceWeight"]
        graphCreator = GraphCreatorFromAdjacencyList(graphFilename,
                                        cellSizeFilename=cellSizeFilename,
                                        useEdgeWeight=useEdgeWeight,
                                        invertEdgeWeight=invertEdgeWeight,
                                        useSharedWallWeight=useSharedWallWeight,
                                        useDistanceWeight=useDistanceWeight)
        currentGraph = graphCreator.GetGraph()
        if maxNormEdgeWeightPerGraph and useEdgeWeight:
            self.normEdgeWeight(edgeProperty="weight")
        return currentGraph

    def normEdgeWeight(self, graph, edgeProperty="weight", inplace=True):
        max = 0
        if inplace is False:
            currentGraph = graph.copy()
        else:
            currentGraph = graph
        for u,v,d in currentGraph.edges(data=True):
            if d[edgeProperty] > max:
                max = d[edgeProperty]
        for u,v,d in currentGraph.edges(data=True):
            d[edgeProperty] /= max
        if inplace is False:
            return currentGraph

    def whereAreElementsIn(self, fullList, elementsOfFullList):
        return [np.where(fullList == elementsOfFullList[i])[0][0] for i in range(len(elementsOfFullList))]

    def GetData(self):
        return self.data

    def GetLabelTable(self):
        return self.labelTable

    def GetFeatureTable(self):
        return self.featureTable

    def SaveLabelTable(self, filenameToSave, sep=","):
        if self.estimateLabels:
            self.labelTable.to_csv(filenameToSave, index=False, sep=sep)
        else:
            print("The labels could not be saved as they were not calculated. self.estimateLabels =", self.estimateLabels)

    def SaveFeatureTable(self, filenameToSave, sep=","):
        if self.estimateFeatures:
            Path(filenameToSave).parent.mkdir(parents=True, exist_ok=True)
            self.featureTable.to_csv(filenameToSave, index=False, sep=sep)
        else:
            print("The features could not be saved as they were not calculated. self.estimateFeatures =", self.estimateFeatures)

def saveFeatureSets(set, dataFolder, plantNames, folderToSave, centralCellsDict=None,
                estimateFeatures=True, estimateLabels=True,
                timePointsPerPlant=5,
                skipEmptyCentrals=False,
                labelsFilename="combinedLabelsChecked.csv"):
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
    if centralCellsDict is None:
        centralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                            "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                            "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                            "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                            "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    else:
        centralCellsDict = centralCellsDict
    myTopologyPredictonDataCreator = TopologyPredictonDataCreator(dataFolder, timePointsPerPlant,
                                            plantNames,
                                            specialGraphProperties=specialGraphProperties,
                                            centralCellsDict=centralCellsDict)
    myTopologyPredictonDataCreator.MakeTrainingData(estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                                                    skipEmptyCentrals=skipEmptyCentrals)
    Path(folderToSave).mkdir(parents=True, exist_ok=True)
    myTopologyPredictonDataCreator.SaveLabelTable(folderToSave+labelsFilename)
    myTopologyPredictonDataCreator.SaveFeatureTable(folderToSave+"combinedFeatures_{}_notnormalised.csv".format(featureProperty))

def getParentLabelingAndParentAndDaughterNetworksOf(plant=3, timeIdx=3, dataFolder="Data/WT/", plantNames=np.asarray(["P1", "P2", "P5", "P6", "P8"])):
    centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    if type(plant) == str:
        plantIdx = np.where(plantNames==plant)[0][0]
    else:
        plantIdx = plant
    plantName = plantNames[plantIdx]
    myTopologyPredictonDataCreator = TopologyPredictonDataCreator(dataFolder, 5, [plantName],centralCellsDict=centralCellsDict, skipFooterOfGeometryFile=4)
    data = myTopologyPredictonDataCreator.GetData()
    parentConnectivityNetwork = data[plantName]["graphCreators"][timeIdx].GetGraph()
    daughterConnectivityNetwork = data[plantName]["graphCreators"][timeIdx+1].GetGraph()
    parentDaughterLabeling = data[plantName]["parentDaugherCellLabeling"][timeIdx]
    return parentDaughterLabeling, parentConnectivityNetwork, daughterConnectivityNetwork

def getValueOf(value="positionalTable", plant=3, timeIdx=3, dataFolder="Data/WT/", plantNames=np.asarray(["P1", "P2", "P5", "P6", "P8"])):
    # value needs to be one of these: ["positionalTable", "graph", "graphCreators", "parentDaugherCellLabeling", "mappedCells", "geometryFilename", "graphFilename", "centralCells"]
    centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    if type(plant) == str:
        plantIdx = np.where(plantNames==plant)[0][0]
    else:
        plantIdx = plant
    plantName = plantNames[plantIdx]
    myTopologyPredictonDataCreator = TopologyPredictonDataCreator(dataFolder, 5, [plantName], centralCellsDict=centralCellsDict, skipFooterOfGeometryFile=4)
    if value == "positionalTable":
        positionalFilename = myTopologyPredictonDataCreator.GetValuesFrom(value="geometryFilename" , plantIdx=plantName, timeIdx=timeIdx, isPlantIdxPlantName=True)
        returnValue = pd.read_csv(positionalFilename, engine="python", skipfooter=4)
    else:
        returnValue = myTopologyPredictonDataCreator.GetValuesFrom(value=value , plantIdx=plantName, timeIdx=timeIdx, isPlantIdxPlantName=True)
    return returnValue

def main():
    # keep in mind to exclude data from tissue to skip (can be identified by column 0 and 1)
    dataFolder = "Data/WT/"
    folderToSave = "Data/WT/topoPredData/diff/"
    plantNames = ["P1", "P2", "P5", "P6", "P8"]
    estimateFeatures = True
    estimateLabels = True
    for set in range(1):
        saveFeatureSets(set=set, dataFolder=dataFolder, plantNames=plantNames,
                        folderToSave=folderToSave,
                        estimateFeatures=estimateFeatures, estimateLabels=estimateLabels)

def mainTest():
    # keep in mind to exclude data from tissue to skip (can be identified by column 0 and 1)
    dataFolder = "Data/ktn/"
    folderToSave = "Data/ktn/topoPredData/"
    plantNames = ["ktnP1", "ktnP2", "ktnP3"]
    estimateFeatures = True
    estimateLabels = True
    centralCellsDict = {"ktnP1": [ [], [3839, 3959] ],
                        "ktnP2": [ [23], [424, 426, 50] ],
                        "ktnP3": [ [29, 199,527], [424, 28, 431] ] }
    for set in range(1, 4):
        saveFeatureSets(set=set, dataFolder=dataFolder, plantNames=plantNames,
                        folderToSave=folderToSave,
                        centralCellsDict=centralCellsDict,
                        estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                        skipEmptyCentrals=True, timePointsPerPlant=3,
                        labelsFilename="combinedLabels.csv")

def mainTestWarningMessage():
    dataFolder = "Data/WT/"
    folderToSave = "Data/WT/Test/"
    plantNames = ["P2"]
    timePointsPerPlant = 5
    centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    estimateFeatures = True
    estimateLabels = True
    specialGraphProperties = None
    featureProperty = "topology"
    skipEmptyCentrals=False
    myTopologyPredictonDataCreator = TopologyPredictonDataCreator(dataFolder, timePointsPerPlant,
                                            plantNames,
                                            specialGraphProperties=specialGraphProperties,
                                            centralCellsDict=centralCellsDict)
    myTopologyPredictonDataCreator.MakeTrainingData(estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                                                    skipEmptyCentrals=skipEmptyCentrals)

if __name__ == '__main__':
    mainTestWarningMessage()
    # mainTest()
