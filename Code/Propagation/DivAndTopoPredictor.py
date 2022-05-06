import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import scipy.stats
import sys

sys.path.insert(0, "./Code/")
sys.path.insert(0, "./Code/Classifiers/")
sys.path.insert(0, "./Code/Feature and Label Creation/")
sys.path.insert(0, "./Code/Predicting/")

from BiologicalFeatureCreatorForNetworkRecreation import BiologicalFeatureCreatorForNetworkRecreation
from DuplicateDoubleRemover import DuplicateDoubleRemover
from FeatureVectorCreator import FeatureVectorCreator
from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from pathlib import Path
from PeripheralCellIdentifier import PeripheralCellIdentifier
from NeighborsOfDividingCellMapper import NeighborsOfDividingCellMapper
from utils import doZNormalise
from TopologyPredictonDataCreatorFromCellIds import saveTopoFeatureSetsFromCellIds

class DivAndTopoPredictor (object):

    # Class to propagate tissue to the next time point
    def __init__(self, divPredModel, topoPredModel, divSampleData, baseFolder,
                 plant, timePoints, divSampleLabel, run=True, seed=42,
                 confirmResultsManually=False, useBioFeaturesForDivPrediction=True,
                 correlateTissues=False, simulateCellDivisions=False,
                 centralCellsList = None,
                 loadPredictionsFromFolder=None, savePredictionsToFolder="Temporary/",
                 topoPredFeatureSet="", normalisationParameterForTopo=None):
        self.divPredModel = divPredModel
        self.topoPredModel = topoPredModel
        self.divSampleData = divSampleData
        self.baseFolder = baseFolder
        self.plant = plant
        self.timePoints = timePoints
        self.divSampleLabel = divSampleLabel
        self.confirmResultsManually = confirmResultsManually
        self.useBioFeaturesForDivPrediction = useBioFeaturesForDivPrediction
        self.centralCellsList = centralCellsList
        self.loadPredictionsFromFolder = loadPredictionsFromFolder
        self.savePredictionsToFolder = savePredictionsToFolder
        self.topoPredFeatureSet = topoPredFeatureSet
        self.normalisationParameterForTopo = normalisationParameterForTopo
        Path(self.savePredictionsToFolder).mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        self.topoPairs = {}
        if run:
            self.setup()
            self.setupDividingCellsAndResultingTopoChanges()
            if self.confirmResultsManually:
                self.printTopoPredLabelsCount()
            if simulateCellDivisions or correlateTissues:
                self.simulateAllCellDivisions()
                if correlateTissues:
                    self.estimateCorrelationsOfNonDivCells()

    def setup(self):
        self.preprocessDivSampleData()
        self.zNormTestDataFromTrainPar(self.plant)
        self.bioMeanPar, self.bioStdPar = self.calcBioFeatureMeanStdToNormalise(self.plant)
        self.allNetworks = self.calcNetworks(self.baseFolder, self.plant, self.timePoints)

    def setupDividingCellsAndResultingTopoChanges(self):
        if self.loadPredictionsFromFolder is None:
            self.predDivEvents()
            if not self.savePredictionsToFolder is None:
                pickle.dump(self.dividingCellsOfTimePoint, open(self.savePredictionsToFolder + "dividingCellsOfTimePoint.pkl", "wb"))
            self.predTopoChanges()
            if not self.savePredictionsToFolder is None:
                pickle.dump(self.topoChangesOfTimePoint, open(self.savePredictionsToFolder + "topoChangesOfTimePoint.pkl", "wb"))
                pickle.dump(self.topoPairs, open(self.savePredictionsToFolder + "topoPairsOfTimePoint.pkl", "wb"))
        else:
            assert type(self.loadPredictionsFromFolder) == str, "loadPredictionsFromFolder needs to be a string, {} != str".format(type(loadPredictionsFromFolder))
            self.dividingCellsOfTimePoint = pickle.load(open(self.loadPredictionsFromFolder+"dividingCellsOfTimePoint.pkl", "rb"))
            self.topoChangesOfTimePoint = pickle.load(open(self.loadPredictionsFromFolder+"topoChangesOfTimePoint.pkl", "rb"))
            self.topoPairs = pickle.load(open(self.loadPredictionsFromFolder+"topoPairsOfTimePoint.pkl", "rb"))

    def preprocessDivSampleData(self):
        duplicateRemover = DuplicateDoubleRemover(self.divSampleData)
        self.duplicateColIdx = duplicateRemover.GetDuplicateColIdx() - 3 - self.useBioFeaturesForDivPrediction
        self.divSampleData = duplicateRemover.GetDoublesReducedTable()

    def zNormTestDataFromTrainPar(self, testPlant="P2"):
        isTestPlant = self.divSampleData.iloc[:, 0] == testPlant
        idxOfTrainPlants = np.where(np.invert(isTestPlant))[0]
        idxOfTestPlant = np.where(isTestPlant)[0]
        self.divSampleData.iloc[idxOfTrainPlants, 3:], self.meanStdPar = doZNormalise(self.divSampleData.iloc[idxOfTrainPlants, 3:], returnParameter=True)
        self.divSampleData.iloc[idxOfTestPlant, 3:] = doZNormalise(self.divSampleData.iloc[idxOfTestPlant, 3:], useParameters=self.meanStdPar)
        # as area is the first feature given but not the first when later used for network feature normalisation
        if self.useBioFeaturesForDivPrediction:
            self.meanStdPar = [self.meanStdPar[0][1:], self.meanStdPar[1][1:]]

    def calcBioFeatureMeanStdToNormalise(self, testPlant, printOut=False):
        topoSampleDataFilename = "Data/WT/topoPredData/diff/manualCentres/bio/combinedFeatures_bio_notnormalised.csv"
        bioTopoSampleData = pd.read_csv(topoSampleDataFilename)
        isTrainValPlant = np.invert(bioTopoSampleData.iloc[:, 0]==testPlant)
        idxOfTissue = np.where(isTrainValPlant)[0]
        bioTrainVal = bioTopoSampleData.iloc[idxOfTissue, 4:]
        _, bioMeanStdPar = doZNormalise(bioTrainVal, returnParameter=True)
        if printOut:
            print(list(bioMeanStdPar[0]), list(bioMeanStdPar[1]))
        return bioMeanStdPar

    def calcNetworks(self, baseFolder, plantName, timePoints, baseFolderExtension="{}/",
                     baseNetworkFilename="cellularConnectivityNetwork{}T{}.csv"):
        allNetworks = []
        if "{}" in baseFolderExtension:
            baseFolder += baseFolderExtension.format(plantName)
        for t in timePoints:
            networkFilename = baseFolder + baseNetworkFilename.format(plantName, t)
            network = GraphCreatorFromAdjacencyList(networkFilename).GetGraph()
            allNetworks.append(network)
        networkFilename = baseFolder + baseNetworkFilename.format(plantName, t+1)
        network = GraphCreatorFromAdjacencyList(networkFilename).GetGraph()
        allNetworks.append(network)
        return allNetworks

    def predDivEvents(self):
        self.dividingCellsOfTimePoint = []
        for t in self.timePoints:
            tissueFeatures = self.selectDivPredFeatures(t)
            cells = self.calcCellId(self.plant, t)
            dividingCells = self.divPredModel.predict(tissueFeatures)
            isCellDividing = dividingCells == 1
            dividingCells = cells[isCellDividing]
            self.dividingCellsOfTimePoint.append(dividingCells)

    def selectDivPredFeatures(self, t, startCol=3):
        idxOfTissue = self.selectIdxOf(self.divSampleData, self.plant, t)
        tissueFeatures = self.divSampleData.iloc[idxOfTissue, startCol:]
        return tissueFeatures

    def selectIdxOf(self, table, plant, timePoint):
        isPlant = table.iloc[:, 0] == plant
        isTime = table.iloc[:, 1] == timePoint
        isTissue = isPlant & isTime
        idxOfTissue = np.where(isTissue)[0]
        return idxOfTissue

    def calcCellId(self, plant, t):
        return self.divSampleData.iloc[self.selectIdxOf(self.divSampleData, self.plant, t), 2].to_numpy()

    def predTopoChanges(self):
        self.topSampleData = self.calcTopoDataOfDividingCells()
        self.topoChangesOfTimePoint = []
        for t in self.timePoints:
            topoFeatures = self.topSampleData[t]
            topoChanges = []
            if len(topoFeatures) > 0:
                topoChanges = self.topoPredModel.predict(topoFeatures)
            self.topoChangesOfTimePoint.append(topoChanges)

    def calcTopoDataOfDividingCells(self, recreateTopoPredFeatures=True):
        selectedCellsOfTissue = {self.plant : self.dividingCellsOfTimePoint}
        if recreateTopoPredFeatures:
            for setIdx in range(7):
                saveTopoFeatureSetsFromCellIds(setIdx, "Data/WT/", [self.plant], self.savePredictionsToFolder,
                                               selectedCellsOfTissue, timePointsPerPlant=5)
        topoPredFeatureTable = pd.read_csv(f"{self.savePredictionsToFolder}{self.topoPredFeatureSet}/combinedFeatures_{self.topoPredFeatureSet}_notnormalised.csv")
        topoPredFeatureTable = self.pruneFeaturesToFitModelFeatures(topoPredFeatureTable)
        if not self.normalisationParameterForTopo is None:
            topoPredFeatureTable.iloc[:, 4:] = doZNormalise(topoPredFeatureTable.iloc[:, 4:], useParameters=self.normalisationParameterForTopo)
        topSampleData = []
        for timePoint, currentFeatures in topoPredFeatureTable.groupby("time point"):
            topSampleData.append(currentFeatures.iloc[:, 4:])
            self.topoPairs[timePoint] = currentFeatures.iloc[:, :4].reset_index(drop=True)
        return topSampleData

    def pruneFeaturesToFitModelFeatures(self, topoPredFeatureTable, allFeatureProperties=["topology", "topologyArea", "topologyWall", "topologyDist"]):
        columns = list(topoPredFeatureTable.columns)
        for i, featureName in enumerate(columns):
            if ".1" in featureName:
                columns[i] = featureName.replace(".1", " "+allFeatureProperties[1])
            elif ".2" in featureName:
                columns[i] = featureName.replace(".2", " "+allFeatureProperties[2])
            elif ".3" in featureName:
                columns[i] = featureName.replace(".3", " "+allFeatureProperties[3])
        originalColumns = list(topoPredFeatureTable.columns)
        topoPredFeatureTable.rename(columns=dict(zip(originalColumns, columns)), inplace=True)
        useGivenFeatureColumns = list(pd.read_csv(f"Results/topoPredData/diff/manualCentres/{self.topoPredFeatureSet}/svm_k2h_combinedTable_l3f0n1c0bal0ex1/normalizedFeatures_train.csv").columns)
        additionalColumnsToKeep = np.asarray(["plant", "time point", "dividing parent cell", "parent neighbor", "cell"])
        topoPredFeatureTable = self.removeNotGivenFeatures(topoPredFeatureTable, useGivenFeatureColumns, additionalColumnsToKeep=additionalColumnsToKeep)
        return topoPredFeatureTable

    def removeNotGivenFeatures(self, table, useGivenFeatureColumns, additionalColumnsToKeep=[],
                               printMissingColumnNames=True,
                               ignoreMissingNames=["dividing parent cell", "parent neighbor", "cell"]):
        useGivenFeatureColumns = np.concatenate([additionalColumnsToKeep, useGivenFeatureColumns])
        columnNames = list(table.columns)
        isFeaturePresent = np.isin(useGivenFeatureColumns, columnNames)
        if printMissingColumnNames:
            missingColumnNames = useGivenFeatureColumns[np.invert(isFeaturePresent)]
            if len(missingColumnNames) != np.sum(np.isin(missingColumnNames, ignoreMissingNames)):
                print(f"The given columns {missingColumnNames} could not be removed as they are not present in the feature table with columns {columnNames} and the columns {ignoreMissingNames} are normaly ignored.")
        useGivenFeatureColumns = useGivenFeatureColumns[isFeaturePresent]
        table = table.loc[:, useGivenFeatureColumns]
        return table

    def simulateAllCellDivisions(self, folderToSave=None, networkBaseName="allPredNetworksDict_{}_T{}.pkl"):
        self.tPlusOneNetworks = []
        for t in self.timePoints:
            nextTimesNetwork = self.applyDivAndTopoPred(t, self.confirmResultsManually)
            self.tPlusOneNetworks.append(nextTimesNetwork)
        if not self.savePredictionsToFolder is None or not folderToSave is None:
            if folderToSave is None:
                folderToSave = self.savePredictionsToFolder
            filenameToSave = folderToSave + networkBaseName.format(self.plant, self.timePoints)
            pickle.dump(self.tPlusOneNetworks, open(filenameToSave, "wb"))

    def applyDivAndTopoPred(self, timePoint, plotAndPrintResults=False):
        oldNetwork = self.allNetworks[timePoint]
        topoPredPairs = self.combineTopoPredAndPairInfos(timePoint)
        orderOfDivCells = self.determineOrderOfInmplementingChanges(topoPredPairs)
        centralCells = None if self.centralCellsList is None else self.centralCellsList[timePoint]
        newNetwork = self.propagateNetwork(oldNetwork, orderOfDivCells, topoPredPairs, centralCells)
        if plotAndPrintResults:
            self.plotAndPrintDivTopoApplication(timePoint, newNetwork, orderOfDivCells)
        return newNetwork

    def combineTopoPredAndPairInfos(self, timePoint):
        topologicalChanges = self.topoChangesOfTimePoint[timePoint]
        topoPairs = self.topoPairs[timePoint]
        predictedLabelTable = pd.DataFrame(topologicalChanges, columns=["labels"])
        topoPredPairs = pd.concat([topoPairs, predictedLabelTable], axis=1)
        return topoPredPairs

    def determineOrderOfInmplementingChanges(self, topoPredPairs):
        dividingCells = np.unique(topoPredPairs.iloc[:, 2])
        np.random.shuffle(dividingCells)
        nrOfDivNeighbors, labelsOfDivNeighbors = self.calcNrAndLabelsOfDivNeighbors(dividingCells, topoPredPairs)
        # divCellsWithDivNeighbor = dividingCells[nrOfDivNeighbors > 0]
        # labels = labelsOfDivNeighbors[nrOfDivNeighbors > 0]
        return dividingCells

    def calcNrAndLabelsOfDivNeighbors(self, dividingCells, topoPredPairs):
        nrOfDivNeighbors = []
        labelsOfDivNeighbors = []
        for d in dividingCells:
            idxOfDivCell = self.getIdxOf(d, topoPredPairs, colIdx=2)
            if len(idxOfDivCell) > 0:
                isDivNeighbor = np.isin(topoPredPairs.iloc[idxOfDivCell, 3], dividingCells)
                nrOfDivNeighbors.append(np.sum(isDivNeighbor))
                if np.any(isDivNeighbor):
                    labelsOfDivNeighbors.append(topoPredPairs.iloc[idxOfDivCell[isDivNeighbor], -1])
                else:
                    labelsOfDivNeighbors.append([])
            else:
                print("The selected dividing cell {} is not inside the topology prediction column dividing cells {}".format(d, np.unique(topoPredPairs.iloc[:, 2])))
        return np.asarray(nrOfDivNeighbors), np.asarray(labelsOfDivNeighbors)

    def getIdxOf(self, entry, table, colIdx=2):
        isEntry = np.isin(table.iloc[:, colIdx], entry)
        return np.where(isEntry)[0]

    def propagateNetwork(self, oldNetwork, orderOfDivCells, topoPredPairs,
                         centralCells=None, addDividedFromPar=True, validateCentralCellsVisually=False):
        newNetwork = oldNetwork.copy()
        nodeIds = list(newNetwork.nodes())
        newCellId = np.max(nodeIds) + 1
        newCentralCells = []
        dividedFromDict = {}
        for i in range(len(orderOfDivCells)):
            divCell = orderOfDivCells[i]
            self.removeDivNeighborEntries(divCell, topoPredPairs)
            self.removeNeighbours(divCell, oldNetwork, newNetwork, orderOfDivCells, i)
            remainingNeighbors = list(newNetwork.neighbors(divCell))
            newNetwork.add_node(newCellId) # divCell is always cell A and newCellId always cell B
            self.reconnectNeighbours(divCell, newCellId, newNetwork,
                                     remainingNeighbors, topoPredPairs)
            if not centralCells is None:
                if divCell in centralCells:
                    newCentralCells.append(newCellId)
            if addDividedFromPar:
                dividedFromDict[divCell] = divCell
                dividedFromDict[newCellId] = divCell
            newCellId += 1
        if not centralCells is None:
            centralCells = np.concatenate([centralCells, newCentralCells])
            wasCentralCellDict = {cell : cell in  centralCells for cell in newNetwork.nodes()}
            nx.set_node_attributes(newNetwork, wasCentralCellDict, name="wasCentralCell")
            if validateCentralCellsVisually:
                self.visualiseCentralCells(newNetwork, propertyName="wasCentralCell")
        if addDividedFromPar:
            for node in newNetwork.nodes():
                if not node in dividedFromDict:
                    dividedFromDict[node] = None
            nx.set_node_attributes(newNetwork, dividedFromDict, name="dividedFrom")
            # self.visualiseCentralCells(newNetwork, propertyName="dividedFrom") # uncomment to viusally check divided cells
        return newNetwork

    def removeDivNeighborEntries(self, divCell, topoPredPairs):
        self.idxOfDivNeighbors = self.getIdxOf(divCell, topoPredPairs, colIdx=3)
        if len(self.idxOfDivNeighbors) > 0:
            topoPredPairs.drop(topoPredPairs.index.to_numpy()[self.idxOfDivNeighbors], axis=0, inplace=True)

    def removeNeighbours(self, divCell, oldNetwork, newNetwork, orderOfDivCells, i):
        divNeighborEdgesToRemove = np.asarray(list(oldNetwork.neighbors(divCell)))
        divNeighborEdgesToRemove = divNeighborEdgesToRemove[np.isin(divNeighborEdgesToRemove, orderOfDivCells[:i], invert=True)]
        for n in divNeighborEdgesToRemove:
            newNetwork.remove_edge(divCell, n)

    def reconnectNeighbours(self, divCell, newCellId, newNetwork,
                            remainingNeighbors, topoPredPairs):
        newNetwork.add_edge(divCell, newCellId)
        for r in remainingNeighbors:
            newNetwork.add_edge(newCellId, r)
        neighborLabelDict = self.determineNeighborLabelDict(divCell, topoPredPairs)
        for neighbor, label in neighborLabelDict.items():
            if label == 0:
                newNetwork.add_edge(divCell, neighbor)
            elif label == 1:
                newNetwork.add_edge(newCellId, neighbor)
            else:
                newNetwork.add_edge(divCell, neighbor)
                newNetwork.add_edge(newCellId, neighbor)

    def visualiseCentralCells(self, network, propertyName="wasCentralCell"):
        wasCentralCellDict = nx.get_node_attributes(network, name=propertyName)
        nodeColors = ["red" if wasCentralCellDict[cell] else "yellow" for cell in wasCentralCellDict.keys()]
        nx.draw(network, node_color=nodeColors, pos=nx.spectral_layout(network), with_labels=True)
        plt.show()

    def determinePossibleProblems(self, topoPredPairs):
        divCells = np.unique(topoPredPairs.iloc[:, 2])
        idxOfDivNeighbors = self.getIdxOf(divCells, topoPredPairs, colIdx=3)

    def determineNeighborLabelDict(self, divCell, topoPredPairs):
        idxOfDivCell = self.getIdxOf(divCell, topoPredPairs)
        neighbors = topoPredPairs.iloc[idxOfDivCell, 3]
        labels = topoPredPairs.iloc[idxOfDivCell, 4]
        return dict(zip(neighbors, labels))

    def estimateCorrelationsOfNonDivCells(self):
        predFeatures, actualFeatures = self.calcFeaturesForNonDivCellsInBoth(self.savePredictionsToFolder)
        np.save(self.savePredictionsToFolder + "actualFeatures.npy", actualFeatures)
        np.save(self.savePredictionsToFolder + "predFeatures.npy", predFeatures)
        correlations = self.correlateFeatures(predFeatures, actualFeatures)
        print(correlations)
        np.save(self.savePredictionsToFolder + "correlations.npy", correlations)

    def calcFeaturesForNonDivCellsInBoth(self, savePredictedNetworksFolder=None):
        allActualFeatures = []
        allPredFeatures = []
        for timePoint in self.timePoints:
             nonDivPredCells = self.calcPredNonDivCells(timePoint)
             nonDivObsCells = self.calcObsNonDivCells(timePoint)
             nonDivCellsBoth = nonDivPredCells[np.isin(nonDivPredCells, nonDivObsCells)]
             # implement tracking of non-dividing cells which are not neighbor of dividing cells
             nonDividingTrackedCells = self.mapObsNonDivCells(timePoint, nonDivCellsBoth)
             actualNetwork = self.allNetworks[timePoint+1]
             predNetwork = self.tPlusOneNetworks[timePoint]
             actualFeatures = self.calcUnweightedTopoFeatures(actualNetwork, list(nonDividingTrackedCells.values()))
             predFeatures = self.calcUnweightedTopoFeatures(predNetwork, list(nonDividingTrackedCells.keys()))
             allActualFeatures.append(actualFeatures)
             allPredFeatures.append(predFeatures)
        allActualFeatures = np.concatenate(allActualFeatures, axis=0)
        allPredFeatures = np.concatenate(allPredFeatures, axis=0)
        return allPredFeatures, allActualFeatures

    def calcPredNonDivCells(self, timePoint):
        network = self.allNetworks[timePoint]
        allCells = np.asarray(network.nodes())
        dividingCells = self.dividingCellsOfTimePoint[timePoint]
        return allCells[np.isin(allCells, dividingCells, invert=True)]

    def calcObsNonDivCells(self, timePoint):
        idxOfTissue = self.selectIdxOf(self.divSampleLabel, self.plant, timePoint)
        idxOfTissueData = self.selectIdxOf(self.divSampleData, self.plant, timePoint)
        cells = self.divSampleLabel.iloc[idxOfTissue, 2]
        cellsData = self.divSampleData.iloc[idxOfTissueData, 2]
        assert np.all(cells==cellsData), "The cells in sampleData of features and labels are not the same np.all(cells==cellsData) != True"
        labels = self.divSampleLabel.iloc[idxOfTissue, -1]
        isNonDividing = labels == 0
        nonDividingCells = cells[isNonDividing]
        return nonDividingCells

    def mapObsNonDivCells(self, timePoint, nonDivCellsToTrack):
        parentConnectivityNetwork = self.getNetworkWithDistanceAtr(timePoint)
        daughterConnectivityNetwork = self.getNetworkWithDistanceAtr(timePoint+1)
        parentLabelFilename = "./Data/WT/{}/parentLabeling{}T{}T{}.csv".format(self.plant, self.plant, timePoint, timePoint+1)
        dividingParentDaughterLabeling = self.calcParentDaughterDictFrom(parentLabelFilename)
        fullParentLabelFilename = "./Data/WT/{}/fullParentLabeling{}T{}T{}.csv".format(self.plant, self.plant, timePoint, timePoint+1)
        fullParentLabeling = self.calcParentDaughterDictFrom(fullParentLabelFilename)
        mapper = NeighborsOfDividingCellMapper(parentConnectivityNetwork,
                                               daughterConnectivityNetwork,
                                               dividingParentDaughterLabeling,
                                               fullParentLabeling)
        mappedCells = mapper.GetMappedCells()
        # pickle.dump(mappedCells, open(self.savePredictionsToFolder + "mappedCells{}T{}.pkl".format(self.plant, timePoint), "wb"))
        # mappedCells = pickle.load(open(self.savePredictionsToFolder + "mappedCells{}T{}.pkl".format(self.plant, timePoint), "rb"))
        allParentCells = [list(mappings.keys()) for mappings in mappedCells.values()]
        allParentCells = np.concatenate(allParentCells)
        missingCellsToMap = nonDivCellsToTrack[np.isin(nonDivCellsToTrack, allParentCells, invert=True)]
        # mappings only contain neighbors
        mappingOfNonDivCells = self.mapp(nonDivCellsToTrack, mappedCells)
        return mappingOfNonDivCells

    def mapp(self, cellsToTrack, mappedCells):
        trackedCells = {}
        for c in cellsToTrack:
            for divCells, mappedNeighbors in mappedCells.items():
                if c in mappedNeighbors:
                    if c in trackedCells:
                        assert trackedCells[c] == mappedNeighbors[c], "The cell {} was mapped to two different daughter cells. {} != {}".format(c, trackedCells[c], mappedNeighbors[c])
                    trackedCells[c] = mappedNeighbors[c]
        return trackedCells

    def getNetworkWithDistanceAtr(self, timePoint):
        cellSizeFilename = "./Data/WT/{}/area{}T{}.csv".format(self.plant, self.plant, timePoint)
        graphFilename = "./Data/WT/{}/cellularConnectivityNetwork{}T{}.csv".format(self.plant, self.plant, timePoint)
        graphCreator = GraphCreatorFromAdjacencyList(graphFilename)
        graphCreator.AddCoordinatesPropertyToGraphFrom(cellSizeFilename)
        return graphCreator.GetGraph()

    def calcParentDaughterDictFrom(self, parentLabelFilename):
        parentLabelingTable = pd.read_csv(parentLabelFilename)
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

    def calcUnweightedTopoFeatures(self, network, selectedCells):
        featureCreator = FeatureVectorCreator(network, selectedCells)
        featureMt = featureCreator.GetFeatureMatrix()
        # duplicateColIdx = self.duplicateColIdx[self.duplicateColIdx<featureMt.shape[1]]
        # featureMt = np.delete(featureMt, duplicateColIdx, axis=1)
        # meanStdPar = [self.meanStdPar[0].to_numpy(), self.meanStdPar[1].to_numpy()]
        # allFeatures = doZNormalise(allFeatures, useParameters=meanStdPar)
        return featureMt

    def correlateFeatures(self, expectedFeatures, observedFeatures):
        correlations = []
        for i in range(expectedFeatures.shape[1]):
            r, p = scipy.stats.pearsonr(expectedFeatures[:, i], observedFeatures[:, i])
            correlations.append(r)
        return correlations

    def printTopoPredLabelsCount(self):
        currentDivCell = None
        predLabels = []
        nrOfDivNeighbors = 0
        dividingCells = np.unique(self.topoPairs[0].iloc[:, 2])
        for i in np.arange(self.topoPairs[0].shape[0]):
            p, t, d, n = self.topoPairs[0].iloc[i, :]
            if currentDivCell != d:
                if not currentDivCell is None:
                    print(currentDivCell, np.unique(predLabels, return_counts=True), "nrOfDivNeighbors", nrOfDivNeighbors)
                currentDivCell = d
                predLabels = []
                nrOfDivNeighbors = 0
            if n in dividingCells:
                nrOfDivNeighbors += 1
            predLabels.append(self.topoChangesOfTimePoint[0][i])
        print(currentDivCell, np.unique(predLabels, return_counts=True), "nrOfDivNeighbors", nrOfDivNeighbors)

    def plotAndPrintDivTopoApplication(self, timePoint, newNetwork, orderOfDivCells):
        oldNetwork = self.allNetworks[timePoint]
        topoPredPairs = self.combineTopoPredAndPairInfos(timePoint)
        startIdx = np.max(list(oldNetwork.nodes())) + 1
        endIdx = np.max(list(newNetwork.nodes()))
        newNodes = np.arange(startIdx, endIdx+1)
        allDividedCells = np.concatenate([orderOfDivCells, newNodes])
        nodes = list(newNetwork)
        nodeColors = np.full(len(nodes), "yellow")
        nodeColors[np.isin(nodes, allDividedCells)] = "blue"
        topoPredPairs = self.combineTopoPredAndPairInfos(timePoint)
        order = []
        arange = np.arange(topoPredPairs.shape[0])
        for divCell in orderOfDivCells:
            isCell = np.isin(topoPredPairs.iloc[:, 2], divCell)
            order.append(arange[isCell])
        order = np.concatenate(order)
        print(topoPredPairs.iloc[order,:].to_string())
        print(orderOfDivCells)
        nx.draw(newNetwork, node_color=nodeColors, pos=nx.spectral_layout(newNetwork), with_labels=True)
        plt.show()

    def calcIfTopologiesAreIdentical(self, observedLocalTopology, predictedLocalTopology):
        return 0

def loadTestModelsAndData(baseFolder, divPredFeatureSet, topoPredFeatureSet=None):
    if topoPredFeatureSet is None:
        topoPredFeatureSet = divPredFeatureSet
    divDataFolder = "{}divEventData/manualCentres/{}/".format(baseFolder, divPredFeatureSet)
    divResultsFolder = "Results/divEventData/manualCentres/{}/svm_k2h_combinedTable_l3f0n1c0bal0ex0/".format(divPredFeatureSet)
    topoResultsFolder = "Results/topoPredData/diff/manualCentres/{}/svm_k2h_combinedTable_l3f0n1c0bal0ex1/".format(topoPredFeatureSet)
    divSampleDataFilename = divDataFolder + "combinedFeatures_{}_notnormalised.csv".format(divPredFeatureSet)
    divSampleLabelFilename = divDataFolder + "combinedLabels.csv"
    divPredModelFilename = divResultsFolder + "testModel.pkl"
    topoPredModelFilename = topoResultsFolder + "testModel.pkl"
    divPredModel = pickle.load(open(divPredModelFilename, "rb"))
    topoPredModel = pickle.load(open(topoPredModelFilename, "rb"))
    divSampleData = pd.read_csv(divSampleDataFilename)
    divSampleLabel = pd.read_csv(divSampleLabelFilename)
    return divPredModel, topoPredModel, divSampleData, divSampleLabel

def findCentralNonPeripheralCells(baseFolder, plantName, timePoint, mostCentralCells,
                                  radiusAroundCenterInMicroM=30,
                                  baseGeometryName="area{}T{}.csv",
                                  basePeripheralLabelName="periphery labels {}T{}.txt"):
    sys.path.insert(0, "./Code/")
    from CellInSAMCenterDecider import CellInSAMCenterDecider
    from utils import convertTextToLabels
    geometryFilename = baseFolder + baseGeometryName.format(plantName, timePoint)
    geometryData = pd.read_csv(geometryFilename, skipfooter=4)
    centralCells = CellInSAMCenterDecider(geometryData, mostCentralCells, centerRadius=radiusAroundCenterInMicroM).GetCentralCells()
    centralCells = centralCells.to_numpy()
    peripheralLabelsFilename = baseFolder + basePeripheralLabelName.format(plantName, timePoint)
    peripheralCellsToRemove = convertTextToLabels(peripheralLabelsFilename,
                                        allLabelsFilename=geometryFilename).GetLabels(onlyUnique=True)
    centralCells = centralCells[np.isin(centralCells, peripheralCellsToRemove, invert=True)]
    return centralCells

def main():
    divPredFeatureSet = "allTopos"
    useBioFeaturesForDivPrediction = divPredFeatureSet=="area" or divPredFeatureSet=="topoAndBio"
    topoPredFeatureSet = "topoAndBio"
    baseFolder = "./Data/WT/"

    normalisationParameterForTopo = np.load(f"Results/topoPredData/diff/manualCentres/{topoPredFeatureSet}/svm_k2h_combinedTable_l3f0n1c0bal0ex1/normalisationParameterMeanAndStd.npy")
    plant = "P2"
    timePoints = [0,1,2,3]
    mostCentralCells = [[392],  [553, 779, 527], [525], [1135]]
    centralCellsList = []
    for i, t in enumerate(timePoints):
        centralCells = findCentralNonPeripheralCells("{}{}/".format(baseFolder, plant), plant, t, mostCentralCells[i])
        centralCellsList.append(centralCells)
    savePredictionsToFolder = f"Results/DivAndTopoApplication/{plant}/"
    loadPredictionsFromFolder = None#savePredictionsToFolder#
    divPredModel, topoPredModel, divSampleData, divSampleLabel = loadTestModelsAndData(baseFolder, divPredFeatureSet, topoPredFeatureSet)
    myDivAndTopoPredictor = DivAndTopoPredictor(divPredModel, topoPredModel,
                                                divSampleData, baseFolder,
                                                plant, timePoints,
                                                divSampleLabel=divSampleLabel,
                                                useBioFeaturesForDivPrediction=useBioFeaturesForDivPrediction,
                                                correlateTissues=True,
                                                simulateCellDivisions=True,
                                                centralCellsList=centralCellsList,
                                                loadPredictionsFromFolder=loadPredictionsFromFolder,
                                                savePredictionsToFolder=savePredictionsToFolder,
                                                topoPredFeatureSet=topoPredFeatureSet,
                                                normalisationParameterForTopo=normalisationParameterForTopo)

def calculateCombinedCorrelations(baseResultsFolder="Results/DivAndTopoApplication/", plantNames=["P2", "P9"]):
    allExpectedFeatures = []
    allPredictedFeatures = []
    for plant in plantNames:
        savePredictionsToFolder = f"{baseResultsFolder}{plant}/"
        actualFeatures = np.load(savePredictionsToFolder + "actualFeatures.npy")
        predFeatures = np.load(savePredictionsToFolder + "predFeatures.npy")
        allExpectedFeatures.append(actualFeatures)
        allPredictedFeatures.append(predFeatures)
    allExpectedFeatures = np.concatenate(allExpectedFeatures, axis=0)
    allPredictedFeatures = np.concatenate(allPredictedFeatures, axis=0)
    correlations = DivAndTopoPredictor(divPredModel=None, topoPredModel=None, divSampleData=None, baseFolder=None,
                 plant=None, timePoints=None, divSampleLabel=None, run=False).correlateFeatures(allExpectedFeatures, allPredictedFeatures)
    np.save(baseResultsFolder + "correlations.npy", correlations)

def propagateAndCorrelateTissues():
    justLoadPredictions = False
    baseResultsFolder = "Results/DivAndTopoApplication/"
    divPredFeatureSet = "allTopos"
    useBioFeaturesForDivPrediction = divPredFeatureSet=="area" or divPredFeatureSet=="topoAndBio"
    topoPredFeatureSet = "topoAndBio"
    baseFolder = "./Data/WT/"
    normalisationParameterForTopo = np.load(f"Results/topoPredData/diff/manualCentres/{topoPredFeatureSet}/svm_k2h_combinedTable_l3f0n1c0bal0ex1/normalisationParameterMeanAndStd.npy")
    plantNames = ["P2", "P9"]
    mostCentralCellsDict = {"P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P9":[[1047, 721, 1048], [7303, 7533], [6735, 7129], [2160, 2228], [7366, 7236]]}
    timePoints = [0,1,2,3]
    divPredModel, topoPredModel, divSampleData, divSampleLabel = loadTestModelsAndData(baseFolder, divPredFeatureSet, topoPredFeatureSet)
    for plant in plantNames:
        centralCellsList = []
        for i, t in enumerate(timePoints):
            centralCells = findCentralNonPeripheralCells("{}{}/".format(baseFolder, plant), plant, t, mostCentralCellsDict[plant][i])
            centralCellsList.append(centralCells)
        savePredictionsToFolder = f"{baseResultsFolder}{plant}/"
        if justLoadPredictions:
            loadPredictionsFromFolder = savePredictionsToFolder
        else:
            loadPredictionsFromFolder = None
        myDivAndTopoPredictor = DivAndTopoPredictor(divPredModel, topoPredModel,
                                                    divSampleData, baseFolder,
                                                    plant, timePoints,
                                                    divSampleLabel=divSampleLabel,
                                                    useBioFeaturesForDivPrediction=useBioFeaturesForDivPrediction,
                                                    correlateTissues=True,
                                                    simulateCellDivisions=True,
                                                    centralCellsList=centralCellsList,
                                                    loadPredictionsFromFolder=loadPredictionsFromFolder,
                                                    savePredictionsToFolder=savePredictionsToFolder,
                                                    topoPredFeatureSet=topoPredFeatureSet,
                                                    normalisationParameterForTopo=normalisationParameterForTopo)
    calculateCombinedCorrelations(baseResultsFolder=baseResultsFolder, plantNames=["P2", "P9"])

if __name__ == '__main__':
    propagateAndCorrelateTissues()
