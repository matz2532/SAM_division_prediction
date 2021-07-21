import copy
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "./Code/PredictTopology/")

from CentralPositionFinder import CentralPositionFinder
from FeatureVectorCreator import FeatureVectorCreator
from FilenameCreator import FilenameCreator
from GraphCenterCellFinder import GraphCenterCellFinder
from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from pathlib import Path
from StandardTableFormater import StandardTableFormater

class DivEventDataCreator (object):

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
        for plantIdx in range(len(self.plantNames)):
            plantData = {}
            filenameIdxOfPlant = np.arange(start=plantIdx, stop=self.totalNrOfSamples,
                                           step=len(self.plantNames))
            graphCreators = self.calcCellNetworksOf(filenameIdxOfPlant)
            plantData["graphCreators"] = graphCreators
            plantData["geometryFilename"] = self.areaFilenames[filenameIdxOfPlant]
            plantData["graphFilename"] = self.connectivityNetworkFilenames[filenameIdxOfPlant]
            plantData["parentDaugherCellLabeling"] = self.parentLabelingFilenames[filenameIdxOfPlant]
            plantName = self.plantNames[plantIdx]
            if not centralCellsDict is None:
                plantData["centralCells"] = centralCellsDict[plantName]
                self.checkExistenceOfNodesInNetworks(centralCellsDict[plantName],
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
            graphCreator = GraphCreatorFromAdjacencyList(filenameCellNetwork,
                                        skipFooter=self.skipFooterOfGeometryFile)
            geometryFilename = self.areaFilenames[filenameIdx]
            graphCreator.AddCoordinatesPropertyToGraphFrom(geometryFilename)
            graphCreators.append(graphCreator)
        return graphCreators

    def checkExistenceOfNodesInNetworks(self, nestedNodesList, networks, plantName):
        for i, nodesList in enumerate(nestedNodesList):
            networkNodes = list(networks[i].GetGraph().nodes())
            isNodeInNetwork = np.isin(nodesList, networkNodes)
            assert np.all(isNodeInNetwork), "In plant {} time {} the nodes {} of {} are not present in the network".format(plantName, i, np.asarray(nodesList)[np.invert(isNodeInNetwork)], nodesList)

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
        allLabels, featureTable = [], []
        for plantIdx in range(len(self.plantNames)):
            for timeIdx in range(self.timePointsPerPlant-1):
                if skipEmptyCentrals:
                    if len(self.centralCellsDict[self.plantNames[plantIdx]][timeIdx]) == 0:
                        continue
                selectedCells = self.determineSelectedCellsOf(plantIdx, timeIdx)
                labelsOfTissue = self.determineLabelsOf(plantIdx, timeIdx, selectedCells)
                if len(labelsOfTissue) > 0:
                    if estimateLabels:
                        allLabels.extend(labelsOfTissue)
                    if estimateFeatures:
                        featureTableOfTissue = self.calcFeaturesOf(plantIdx, timeIdx, selectedCells)
                        featureTable.append(featureTableOfTissue)
            if self.printProgress:
                print("Finished plant {}".format(self.plantNames[plantIdx]))
        if estimateLabels:
            columnNames = ["plant", "time point", "cell", "label"]
            self.labelTable = pd.DataFrame(allLabels, columns=columnNames)
        if estimateFeatures:
            self.featureTable = pd.concat(featureTable)

    def determineSelectedCellsOf(self, plantIdx, timeIdx):
        # select cells inside radius as well as not being on the edge of the graph
        parentGraph = self.GetValuesFrom("graph", plantIdx, timeIdx)
        plantName = self.plantNames[plantIdx]
        peripheralLabelFilename = self.dataFolder + "{}/periphery labels {}T{}.txt".format(plantName, plantName, timeIdx)
        geometryFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx)
        centerCellLabels = self.GetValuesFrom("centralCells", plantIdx, timeIdx)
        selectedCells = GraphCenterCellFinder(self.dataFolder, self.plantNames[plantIdx], timeIdx, self.centerRadius, parentGraph,
                     peripheralLabelFilename, geometryFilename, centerCellLabels).defineValidLabels()
        selectedCells = selectedCells.to_numpy()
        return np.sort(selectedCells)

    def GetValuesFrom(self, value, plantIdx, timeIdx, isPlantIdxPlantName=False):
        validValues = ["graph", "graphCreators", "parentDaugherCellLabeling",
                       "geometryFilename", "graphFilename", "centralCells"]
        assert value in validValues, "The value {} you choose needs to be one of the following values {}".format(value, validValues)
        if isPlantIdxPlantName:
            assert plantIdx in self.data, "plantIdx needs to be a plant name (key) in self.data if isPlantIdxPlantName is True. {} is not in {}".format(plantIdx, self.data.keys())
            plantName = plantIdx
        else:
            assert type(plantIdx) == int, "plantIdx needs to be an integer. {} is of type {} != int".format(plantIdx, type(plantIdx))
            plantName = self.plantNames[plantIdx]
        plantData = self.data[plantName]
        if value == "graph":
            assert timeIdx < len(plantData["graphCreators"]), "Time index is out of range. {} < {}".format(timeIdx, len(plantData[value][timeIdx]))
            return plantData["graphCreators"][timeIdx].GetGraph()
        else:
            assert value in plantData, "The key {} is not present in as a key: {}.".format(value, plantData.keys())
            assert timeIdx < len(plantData[value]), "Time index is out of range. {} < {}".format(timeIdx, len(plantData[value][timeIdx]))
            return plantData[value][timeIdx]

    def determineLabelsOf(self, plantIdx, timeIdx, selectedCells):
        self.myStandardTableFormater.SetPlantNameByIdx(plantIdx)
        self.myStandardTableFormater.SetTimePoint(timeIdx)
        self.parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx)
        self.parentDaugherCellLabeling = self.GetValuesFrom("parentDaugherCellLabeling", plantIdx, timeIdx)
        parentLabelingTable = pd.read_csv(self.parentDaugherCellLabeling, sep=self.sep)
        self.parentDaugherCellLabeling = self.calcParentDaughterDictFrom(parentLabelingTable)
        labelDataOfPlantTissue = []
        dividingParentCells = np.asarray(list(self.parentDaugherCellLabeling.keys()))
        if len(dividingParentCells) > 0:
            for cell in selectedCells:
                if cell in dividingParentCells:
                    label = 1
                else:
                    label = 0
                labelData = [cell, label]
                out = self.myStandardTableFormater.GetProperStandardFormatWith(labelData)
                labelDataOfPlantTissue.append(out)
        return labelDataOfPlantTissue

    def calcParentDaughterDictFrom(self, parentLabelingTable):
        parentDaughterDict = {}
        dividingParentCell = parentLabelingTable.iloc[:, 1].to_numpy()
        daughterLabels = parentLabelingTable.iloc[:, 0].to_numpy()
        indexes = np.unique(dividingParentCell, return_index=True)[1]
        for index in sorted(indexes):
            currentLabel = dividingParentCell[index]
            isCurrentLabel = dividingParentCell == currentLabel
            daughterCellsOfDividingCell = daughterLabels[isCurrentLabel]
            parentDaughterDict[currentLabel] = daughterCellsOfDividingCell
        return parentDaughterDict

    def calcFeaturesOf(self, plantIdx, timeIdx, selectedCells):
        self.myStandardTableFormater.SetPlantNameByIdx(plantIdx)
        self.myStandardTableFormater.SetTimePoint(timeIdx)
        standardFormat = self.myStandardTableFormater.GetProperStandardFormatWithEveryEntryIn(selectedCells)
        sourcesOfFeatures = pd.DataFrame(standardFormat, columns=["plant", "time point", "cell"])
        if self.onlyAra:
            uniqueFeatureDF = self.calcAreaFeatureOf(plantIdx, timeIdx, selectedCells)
        else:
            uniqueFeatureDF = self.calcFeatureTable(plantIdx, timeIdx, selectedCells)
        combinedFeatureDF = pd.concat([sourcesOfFeatures, uniqueFeatureDF], axis=1)
        return combinedFeatureDF

    def calcAreaFeatureOf(self, plantIdx, timeIdx, selectedCells):
        cellSizeFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx)
        sizeTable = pd.read_csv(cellSizeFilename, sep=self.sep, engine="python",
                                skipfooter=self.skipFooterOfGeometryFile)
        allCells = sizeTable.iloc[:, 0]
        allAreas = sizeTable.iloc[:, 2]
        sizeFeature = []
        for cell in selectedCells:
            isCell = np.isin(allCells, cell)
            assert np.sum(isCell) == 1, "The cell {} either doesn' exist np.sum(isCell) == 0 or it exists multiple times np.sum(isCell) > 1 in {}; np.sum(isCell) == {}".format(cell, cellSizeFilename, np.sum(isCell))
            area = allAreas[isCell]
            sizeFeature.append(area.to_numpy())
        featureDF = pd.DataFrame(sizeFeature, columns=["area"])
        featureDF.index = np.arange(len(selectedCells))
        return featureDF

    def calcFeatureTable(self, plantIdx, timeIdx, selectedCells):
        if self.specialGraphProperties is None:
            parentConnectivityNetwork = self.GetValuesFrom("graph", plantIdx, timeIdx)
        else:
            parentConnectivityNetwork = self.calcSpecialGraph(plantIdx, timeIdx)
            cellSizeFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx)
        myFeatureVectorCreator = FeatureVectorCreator(parentConnectivityNetwork,
                                                      allowedLabels=selectedCells,
                                                      zNormaliseFeaturesPerTissue=self.zNormaliseFeaturesPerTissue)
        featureDF = myFeatureVectorCreator.GetFeatureMatrixDataFrame()
        featureDF.index = np.arange(len(selectedCells))
        return featureDF

    def calcSpecialGraph(self, plantIdx, timeIdx):
        cellSizeFilename = self.GetValuesFrom("geometryFilename", plantIdx, timeIdx)
        graphFilename = self.GetValuesFrom("graphFilename", plantIdx, timeIdx)
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

    def GetData(self):
        return self.data

    def GetLabelTable(self):
        return self.labelTable

    def GetFeatureTable(self):
        return self.featureTable

    def SaveLabelTable(self, filenameToSave, sep=","):
        if not self.labelTable is None:
            self.labelTable.to_csv(filenameToSave, index=False, sep=sep)
        else:
            print("The labels could not be saved as they were not calculated. self.labelTable = {}".format(self.labelTable))

    def SaveFeatureTable(self, filenameToSave, sep=","):
        if not self.featureTable is None:
            self.featureTable.to_csv(filenameToSave, index=False, sep=sep)
        else:
            print("The features could not be saved as they were not calculated. self.featureTable =", self.featureTable)

def combineFeatures(folderToSave, featurePropertyName="", includeBioFeatures=False):
    allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist"]
    if includeBioFeatures:
        allFeatureProperties.insert(0, "area")
    allTables = []
    for i, featureProperty in enumerate(allFeatureProperties):
        filenameToLoad = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featureProperty, featureProperty)
        table = pd.read_csv(filenameToLoad)
        if i != 0:
            table = table.drop(columns=["plant", "time point", "cell"], axis=1)
        allTables.append(table)
    allTables = pd.concat(allTables, axis=1)
    if featurePropertyName != "":
        folderToSave += "{}/".format(featurePropertyName)
        filenameToSave = folderToSave + "combinedFeatures_{}_notnormalised.csv".format(featurePropertyName)
    else:
        filenameToSave = folderToSave + "combinedLabels_notnormalised.csv"
    Path(folderToSave).mkdir(parents=True, exist_ok=True)
    allTables.to_csv(filenameToSave, index=False)

def saveLowCorrelationOf(folderToSave, featurePropertyToSave,
                         featurePropertyToLoad, rThreshold, propertyCorAgainst="area"):
    import scipy.stats
    filenameToLoad = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featurePropertyToLoad, featurePropertyToLoad)
    table = pd.read_csv(filenameToLoad)
    standardFormat = table.iloc[:, :3]
    valuesToCorrelate = table.iloc[:, 3:]
    filenameToLoad = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(propertyCorAgainst, propertyCorAgainst)
    corAgainst = pd.read_csv(filenameToLoad).iloc[:, -1]
    nrOfCol = valuesToCorrelate.shape[1]
    correlations = np.zeros(nrOfCol)
    for i in range(nrOfCol):
        r, p = scipy.stats.pearsonr(corAgainst, valuesToCorrelate.iloc[:, i])
        correlations [i] = r
    columnsToKeep = np.where(correlations < rThreshold)[0]
    reducedValues = valuesToCorrelate.iloc[:, columnsToKeep]
    reducedTable = pd.concat([standardFormat, reducedValues], axis=1)
    folderToSave = folderToSave + "{}/".format(featurePropertyToSave)
    Path(folderToSave).mkdir(parents=True, exist_ok=True)
    filenameToSave = folderToSave + "combinedFeatures_{}_notnormalised.csv".format(featurePropertyToSave)
    reducedTable.to_csv(filenameToSave, index=False)

def saveFeatureSets(set, dataFolder, plantNames, folderToSave, centralCellsDict=None,
                estimateFeatures=True, estimateLabels=True,
                useManualCentres=False, timePointsPerPlant=5, skipEmptyCentrals=False):
    import shutil
    specialGraphProperties = None
    featureProperty = "topology"
    onlyAra = False
    if useManualCentres or not centralCellsDict is None:
        folderToSave += "manualCentres/"
        if centralCellsDict is None:
            centralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                            "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                            "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                            "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                            "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
        else:
            centralCellsDict = centralCellsDict
    else:
        centralCellsDict = None
    labelsFilenameToCopy = folderToSave+"{}/combinedLabels.csv".format(featureProperty)
    if set == 1:
        # for area
        featureProperty = "topologyArea"
        Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
        if estimateLabels:
            shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        estimateLabels = False
        specialGraphProperties = {"useEdgeWeight":True,
                                  "invertEdgeWeight": True,
                                  "useSharedWallWeight": False,
                                  "useDistanceWeight": False,
                                  "maxNormEdgeWeightPerGraph": False}
    elif set == 2:
        # for Wall
        featureProperty = "topologyWall"
        Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
        if estimateLabels:
            shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        estimateLabels = False
        specialGraphProperties = {"useEdgeWeight":True,
                                  "invertEdgeWeight": False,
                                  "useSharedWallWeight": True,
                                  "useDistanceWeight": False,
                                  "maxNormEdgeWeightPerGraph": False}
    elif set == 3:
        # for distance
        featureProperty = "topologyDist"
        Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
        if estimateLabels:
            shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        estimateLabels = False
        specialGraphProperties = {"useEdgeWeight":True,
                                  "invertEdgeWeight": False,
                                  "useSharedWallWeight": False,
                                  "useDistanceWeight": True,
                                  "maxNormEdgeWeightPerGraph": False}
    elif set == 4:
        featureProperty = "area"
        Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
        if estimateLabels:
            shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        estimateLabels = False
        onlyAra = True
    elif set == 5:
        featureProperty = "allTopos"
        Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
        if estimateLabels:
            shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        combineFeatures(folderToSave, featureProperty)
        return None
    elif set == 6:
        featureProperty = "topoAndBio"
        Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
        if estimateLabels:
            shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        combineFeatures(folderToSave, featureProperty, includeBioFeatures=True)
        return None
    elif set == 7:
        for rThreshold in [0.5, 0.7]:
            featurePropertyToLoad = "allTopos"
            featureProperty = "lowCor{}".format(rThreshold)
            Path(folderToSave+"{}/".format(featureProperty)).mkdir(parents=True, exist_ok=True)
            saveLowCorrelationOf(folderToSave, featureProperty, featurePropertyToLoad, rThreshold)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
        return None
    myDivEventDataCreator = DivEventDataCreator(dataFolder, timePointsPerPlant,
                                            plantNames,
                                            specialGraphProperties=specialGraphProperties,
                                            centralCellsDict=centralCellsDict,
                                            onlyAra=onlyAra)
    myDivEventDataCreator.MakeTrainingData(estimateFeatures=True, estimateLabels=estimateLabels,
                                           skipEmptyCentrals=skipEmptyCentrals)
    trainingData = myDivEventDataCreator.GetFeatureTable()
    labelData = myDivEventDataCreator.GetLabelTable()
    print("trainingData", trainingData)
    folderToSave += featureProperty + "/"
    Path(folderToSave).mkdir(parents=True, exist_ok=True)
    myDivEventDataCreator.SaveFeatureTable(folderToSave+"combinedFeatures_{}_notnormalised.csv".format(featureProperty))
    myDivEventDataCreator.SaveLabelTable(folderToSave+"combinedLabels.csv")

def mainCreateWT():
    # keep in mind to exclude data from tissue to skip (can be identified by column 0 and 1)
    dataFolder = "Data/WT/"
    folderToSave = "Data/WT/divEventData/"
    plantNames = ["P1", "P2", "P5", "P6", "P8"]
    estimateFeatures = True
    estimateLabels = True
    useManualCentres = True
    for set in range(7):
        print("set: ", set)
        saveFeatureSets(set=set, dataFolder=dataFolder, plantNames=plantNames,
                        folderToSave=folderToSave,
                        estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                        useManualCentres=useManualCentres)

def mainCreateMutantFeatureSets():
    # keep in mind to exclude data from tissue to skip (can be identified by column 0 and 1)
    dataFolder = "Data/ktn/"
    folderToSave = "Data/ktn/divEventData/"
    plantNames = ["ktnP1", "ktnP2", "ktnP3"]
    estimateFeatures = True
    estimateLabels = True
    useManualCentres = True
    centralCellsDict = {"ktnP1": [ [], [3839, 3959] ],
                        "ktnP2": [ [23], [424, 426, 50] ],
                        "ktnP3": [ [29, 199,527], [424, 28, 431] ] }
    for set in range(7):
        print("set: ", set)
        saveFeatureSets(set=set, dataFolder=dataFolder, plantNames=plantNames,
                        folderToSave=folderToSave,
                        estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                        useManualCentres=useManualCentres,
                        centralCellsDict=centralCellsDict,
                        skipEmptyCentrals=True,
                        timePointsPerPlant=3)

if __name__ == '__main__':
    # mainCreateWT()
    mainCreateMutantFeatureSets()
