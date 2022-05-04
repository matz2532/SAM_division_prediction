import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import scipy
import scipy.stats
import sys

sys.path.insert(0, "./Code/")
sys.path.insert(0, "./Code/Feature and Label Creation/")
from DivAndTopoPredictor import DivAndTopoPredictor
from utils import convertParentLabelingTableToDict
from pathlib import Path
from PeripheralCellIdentifier import PeripheralCellIdentifier

class LocalTopologyPredictionComparer (DivAndTopoPredictor):

    def __init__(self, divPredModel, topoPredModel, divSampleData, baseFolder,
                 plant, timePoints, divSampleLabel, loadPredictionsFromFolder):
        super().__init__(divPredModel, topoPredModel, divSampleData, baseFolder,
                       plant, timePoints, divSampleLabel,
                       loadPredictionsFromFolder=loadPredictionsFromFolder,
                       correlateTissues=False, simulateCellDivisions=False)

    def CompareSingleDividingNeighbourhoods(self, topologicalChangesTable, resultsFolder=None):
        self.topologicalChangesTable = topologicalChangesTable
        self.resultsFolder = resultsFolder
        nrOfDiviations = []
        nrOfNeighbours = []
        for t in self.timePoints:
            predAndObsDivCells = self.prepareDataForCurrentTissue(t)
            for dividingCell in predAndObsDivCells:
                diviationsAndTotal = self.compareDividingCellsTopology(dividingCell, t)
                if not diviationsAndTotal is None:
                    diviations = diviationsAndTotal[0]
                    neighours = diviationsAndTotal[1]
                    nrOfDiviations.append(diviations)
                    nrOfNeighbours.append(neighours)
        np.save(self.loadPredictionsFromFolder+"nrOfNeighbours.npy", nrOfNeighbours)
        np.save(self.loadPredictionsFromFolder+"nrOfDiviations.npy", nrOfDiviations)
        return nrOfDiviations, nrOfNeighbours

    def prepareDataForCurrentTissue(self, timePoint):
        self.currentTissueName = "{} {}".format(self.plant, timePoint)
        self.currentParentTissueNetwork, self.currentDaughterTissueNetwork = self.calcNetworks(self.baseFolder, self.plant, [timePoint])
        predAndObsDivCells = self.findPredictedAndObservedDividingCells(timePoint)
        self.currentParentLabeling = self.loadParentLabelingOfTissue(timePoint, predAndObsDivCells)
        self.currentPredictedTopoChangesTable = self.mergeCurrentPredictedTopoChangesTable(timePoint)
        return predAndObsDivCells

    def findPredictedAndObservedDividingCells(self, timePoint):
        idxOfObsTissue = self.selectIdxOf(self.topologicalChangesTable, self.plant, timePoint)
        observedDividingCells = np.unique(self.topologicalChangesTable.iloc[idxOfObsTissue, 2])
        predictedDividingCells = self.dividingCellsOfTimePoint[timePoint]
        isObsCellPresentInPred = np.isin(observedDividingCells, predictedDividingCells)
        predAndObsDivCells = observedDividingCells[isObsCellPresentInPred]
        return predAndObsDivCells

    def loadParentLabelingOfTissue(self, timePoint, cellsToChekForPresence=[], sep=","):
        filename = self.baseFolder + "{}/fullParentLabeling{}T{}T{}.csv".format(self.plant, self.plant, timePoint, timePoint+1)
        parentLabelingTable = pd.read_csv(filename, sep=sep)
        parentLabelingDict = convertParentLabelingTableToDict(parentLabelingTable)
        isCellPresent = np.isin(cellsToChekForPresence, list(parentLabelingDict.keys()))
        areAllCellsPresent = np.all(isCellPresent)
        assert areAllCellsPresent, "The cells {} are not present in the parent labeling file {} only {} cells are present.".format(cellsToChekForPresence[isCellPresent], filename, list(parentLabelingDict.keys()))
        return parentLabelingDict

    def mergeCurrentPredictedTopoChangesTable(self, timePointIdx):
        predictedDividingCells = self.dividingCellsOfTimePoint[timePointIdx]
        topoPairs = self.topoPairs[timePointIdx]
        predictedTopoClasses = self.topoChangesOfTimePoint[timePointIdx]
        nrRows = len(predictedTopoClasses)
        plantNameCol = np.full(nrRows, self.plant)
        timePointCol = np.full(nrRows, timePointIdx)
        dividingCells, theirNeighbours = topoPairs.iloc[:, 2].to_numpy(), topoPairs.iloc[:, 3].to_numpy()
        currentPredictedTopoChanges = np.concatenate([[dividingCells], [theirNeighbours], [predictedTopoClasses]], axis=0)
        currentPredictedTopoChangesTable = pd.DataFrame(currentPredictedTopoChanges.T)
        return currentPredictedTopoChangesTable

    def compareDividingCellsTopology(self, dividingCell, timePoint):
        observedLocalTopology = self.extractLocalTopologyOf(dividingCell, timePoint)
        predictedLocalTopology = self.extractLocalTopologyOf(dividingCell, timePoint, useObservedTopology=False)
        if not observedLocalTopology is None and not predictedLocalTopology is None:
            numberOfDifferences = self.calcDifferencesInTopologies(observedLocalTopology, predictedLocalTopology)
            if not self.resultsFolder is None:
                folder = self.resultsFolder + "{}/".format(self.currentTissueName)
                Path(folder).mkdir(parents=True, exist_ok=True)
                filenameToSave = "{}dividingCell_{}_obs_vs_pred_LocanTopology.png".format(folder, dividingCell)
            else:
                filenameToSave = None
            self.plotTopologies([observedLocalTopology, predictedLocalTopology], filenameToSave=filenameToSave)
            neighbours = observedLocalTopology.number_of_nodes() - 2
            return numberOfDifferences, neighbours

    def extractLocalTopologyOf(self, dividingCell, timePoint, useObservedTopology=True,
                               cellAName="A", cellBName="B"):
        neighbours = list(self.currentParentTissueNetwork.neighbors(dividingCell))
        subgraph = nx.Graph(self.currentParentTissueNetwork.subgraph(neighbours))
        if useObservedTopology:
            topologicalClasses = self.selectTopologcialClassOfNeighbouring(neighbours, dividingCell,
                                                self.plant, timePoint, selectObserved=True)
        else:
            topologicalClasses = self.selectTopologcialClassOfNeighbouring(neighbours, dividingCell,
                                                self.plant, timePoint, selectObserved=False,
                                                useAllInidces=True, dividingCellsIdx=0, parentNeighbourIdx=1)
        if topologicalClasses is None:
            return None
        subgraph.add_node(cellAName)
        subgraph.add_node(cellBName)
        edgesToAdd = self.createEdgeTouplesBasedOnClasses(neighbours, topologicalClasses, cellAName=cellAName, cellBName=cellBName)
        subgraph.add_edges_from(edgesToAdd)
        subgraph.add_edge(cellAName, cellBName)
        neighbourClassDict = dict(zip(neighbours, topologicalClasses))
        colours = self.setColoursBasedOnClassesFor(subgraph, neighbourClassDict, colourMapping={0:"#2EF8F9", 1:"#F59C24", 2:"#E550AB", "leftOver":"#A6831C"})
        return subgraph

    def selectTopologcialClassOfNeighbouring(self, listOfCells, dividingCell, plantName, timePoint,
                                             selectObserved=True, useAllInidces=False,
                                             dividingCellsIdx=2, parentNeighbourIdx=3,
                                             classIdx=-1):
        if selectObserved:
            topologicalChangesTable = self.topologicalChangesTable
        else:
            topologicalChangesTable = self.currentPredictedTopoChangesTable
        topologicalClasses = []
        if useAllInidces:
            allCellsIndices = np.arange(len(topologicalChangesTable))
        else:
            allCellsIndices = self.selectIdxOf(topologicalChangesTable, plantName, timePoint)
        possibleDividingCells = topologicalChangesTable.iloc[allCellsIndices, dividingCellsIdx].to_numpy()
        neighbouringCells = topologicalChangesTable.iloc[allCellsIndices, parentNeighbourIdx].to_numpy()
        classesOfTissuesCells = topologicalChangesTable.iloc[allCellsIndices, classIdx].to_numpy()
        isDividingCell = np.isin(possibleDividingCells, dividingCell)
        for cell in listOfCells:
            isNeighbour = np.isin(neighbouringCells, cell)
            isSelectedCell = isNeighbour & isDividingCell
            # assert np.sum(isSelectedCell) == 1, "The selected cell {} is not present exactly once with the dividing cell {} for {} {}, {} != 1".format(cell, dividingCell, self.currentTissueName, selectObserved, np.sum(isSelectedCell))
            if np.sum(isSelectedCell) == 1:
                classOfCell = classesOfTissuesCells[isSelectedCell][0]
                topologicalClasses.append(classOfCell)
            else:
                print("The selected cell {} is not present exactly once with the dividing cell {} for {} {}, {} != 1".format(cell, dividingCell, self.currentTissueName, selectObserved, np.sum(isSelectedCell)))
                return None
        return topologicalClasses

    def createEdgeTouplesBasedOnClasses(self, connectWithNodes, topologicalClasses,
                                        cellAName="A", cellBName="B"):
        edgesToAdd = []
        for node, topologicalClass in zip(connectWithNodes, topologicalClasses):
            if topologicalClass == 0:
                edgesToAdd.append([cellAName, node])
            elif topologicalClass == 1:
                edgesToAdd.append([cellBName, node])
            elif topologicalClass == 2:
                edgesToAdd.append([cellAName, node])
                edgesToAdd.append([cellBName, node])
        return edgesToAdd

    def setColoursBasedOnClassesFor(self, subgraph, neighbourClassDict,
                                    colourMapping={0:"#2EF8F9", 1:"#F59C24", 2:"#E550AB", "leftOver":"#A6831C"}):
        colouring = {}
        for node in subgraph.nodes():
            if node in neighbourClassDict:
                topoClass = neighbourClassDict[node]
            else:
                topoClass = "leftOver"
            colour = colourMapping[topoClass]
            colouring[node] = colour
        nx.set_node_attributes(subgraph, name="colouring", values=colouring)

    def calcDifferencesInTopologies(self, graph1, graph2, nodesToExclude=["A", "B"]):
        differencesPerTopology = 0
        for node in graph1.nodes():
            if not node in nodesToExclude:
                neighboursOfGraph1 = list(graph1.neighbors(node))
                neighboursOfGraph2 = list(graph2.neighbors(node))
                if neighboursOfGraph1 != neighboursOfGraph2:
                    differencesPerTopology += 1
        return differencesPerTopology

    def plotTopologies(self, listOfNetworks, showLabels=True, filenameToSave=None, showPlot=True,
                       colourNodesByClass=True, useAttrPosition=True,
                       displaySubtitles=True):
        nrOfNetworks = len(listOfNetworks)
        fig, ax = plt.subplots(1, nrOfNetworks)
        for i in range(nrOfNetworks):
            currentNetwork = listOfNetworks[i]
            if colourNodesByClass:
                colour = list(nx.get_node_attributes(currentNetwork, "colouring").values())
            else:
                colour = "#1f78b4"
            if useAttrPosition:
                if i==0:
                    pos = nx.spectral_layout(currentNetwork)
            else:
                pos = nx.spectral_layout(currentNetwork)
            nx.draw(currentNetwork, ax=ax[i], pos=pos,
                    with_labels=showLabels, node_color=colour)
            if displaySubtitles:
                if i == 0:
                    title = "observed tissue"
                else:
                    title = "expecteded tissue"
                ax[i].set_title(title)
        if filenameToSave:
            plt.savefig(filenameToSave, bbox_inches="tight")
            plt.close()
        elif showPlot:
            plt.show()

def compareLocalTopologyPrediction():
    from DivAndTopoPredictor import loadTestModelsAndData
    # apply topo pred and compare it with observed tissue by only looking at how dividing cell and its neighbours change
    # merge div neighbours
    # compare with two images per figure
    divPredFeatureSet = "allTopos"
    useBioFeaturesForDivPrediction = divPredFeatureSet=="area" or divPredFeatureSet=="topoAndBio"
    topoPredFeatureSet = "topoAndBio"
    baseFolder = "./Data/WT/"
    baseResultsFolder = "Results/DivAndTopoApplication/"
    plantNames = ["P2", "P9"]
    timePoints = [0,1,2,3]
    allNumberOfNeighbours, allErrorsPerTopo = [], []
    for plant in plantNames:
        imageResultsFolder = f"{baseResultsFolder}{plant}/VisualisingPropagation/"
        loadPredictionsFromFolder = f"{baseResultsFolder}{plant}/"
        topologicalChangesTable = pd.read_csv("./Data/WT/topoPredData/diff/manualCentres/topoAndBio/combinedLabels.csv")
        divPredModel, topoPredModel, divSampleData, divSampleLabel = loadTestModelsAndData(baseFolder, divPredFeatureSet, topoPredFeatureSet)
        myComparer = LocalTopologyPredictionComparer(divPredModel=None, topoPredModel=None,
                                                    divSampleData=divSampleData, baseFolder=baseFolder,
                                                    plant=plant, timePoints=timePoints,
                                                    divSampleLabel=divSampleLabel,
                                                    loadPredictionsFromFolder=loadPredictionsFromFolder)
        errorsPerTopo, numberOfNeighbours = myComparer.CompareSingleDividingNeighbourhoods(topologicalChangesTable, resultsFolder=imageResultsFolder)
        allNumberOfNeighbours.append(numberOfNeighbours)
        allErrorsPerTopo.append(errorsPerTopo)
    allNumberOfNeighbours = np.concatenate(allNumberOfNeighbours)
    allErrorsPerTopo = np.concatenate(allErrorsPerTopo)
    np.save(baseResultsFolder+"nrOfNeighbours.npy", numberOfNeighbours)
    np.save(baseResultsFolder+"nrOfDiviations.npy", errorsPerTopo)

def calcRandomDistributionOfErrorsFor(numberOfNeighbours, repetitions=1000,
                                      percentage=True, columnNames=None):
    randomDistributions = []
    for n in numberOfNeighbours:
        numberOfErrors = np.random.binomial(size=repetitions, n=n, p=2/3)
        if percentage:
            numberOfErrors = numberOfErrors / n
        randomDistributions.append(numberOfErrors)
    randomDistributions = np.concatenate(randomDistributions)
    return pd.DataFrame(randomDistributions, columns=columnNames)

def plotPercentageCorrectTopologies(baseResultsFolder="Results/DivAndTopoApplication/",
                                    folderToSave=None, plotName="topology prediction density plot.png",
                                    percentage=True, fontSize=20):
    numberOfNeighbours = np.load(baseResultsFolder+"nrOfNeighbours.npy")
    errorsPerTopo = np.load(baseResultsFolder+"nrOfDiviations.npy")
    randomDistributions = calcRandomDistributionOfErrorsFor(numberOfNeighbours, percentage=percentage, columnNames=["random"])
    errorsPerTopo = np.asarray(errorsPerTopo)
    numberOfNeighbours = np.asarray(numberOfNeighbours)
    if percentage:
        errorsPerTopo = errorsPerTopo / numberOfNeighbours
    d = pd.DataFrame(errorsPerTopo, columns=["predicted"])
    plt.rcParams.update({"font.size": fontSize})
    fig, ax = plt.subplots()
    if percentage:
        d = 100*(1 - d)
        randomDistributions = 100*(1 - randomDistributions)
        plt.xlim((0, 1))
        bw_method = 0.3
    else:
        bw_method = .5
        plt.xlim((0, 8))
    ksTest = scipy.stats.ks_2samp(d.iloc[:, 0].to_numpy(), randomDistributions.iloc[:, 0].to_numpy())
    distributionResultsText = f"Distribution {ksTest} from {baseResultsFolder}"
    axTwin = ax.twinx()
    d.plot.hist(legend=False, density=False, ax=axTwin, zorder=1)
    d.plot.kde(legend=False, ax=ax, bw_method=bw_method, lw=4, color="black", zorder=2)
    d.plot.kde(legend=False, ax=ax, bw_method=bw_method, lw=2, zorder=2)
    randomDistributions.plot.kde(legend=False, ax=ax, bw_method=bw_method, lw=2, zorder=2)
    ax.set_zorder(axTwin.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel("Probability")
    ax.set_facecolor("#d8dcd6")
    axTwin.set_ylabel("Count")
    axTwin.set_ylim(0, 17)
    axTwin.set_yticks(range(0,16,5))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    legendHanldes = [mpatches.Patch(color=colors[0], label="predicted"), mpatches.Patch(color=colors[1], label="random")]
    plt.legend(handles=legendHanldes, prop={"size": fontSize-8})
    if percentage:
        plt.xlim((0, 100))
    else:
        plt.xlim((0, 8))
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    axTwin.spines["top"].set_visible(False)
    ax.set_xlabel("Percentage of correctly labeled neighbours\nper local topology")
    if not folderToSave is None:
        plt.savefig(folderToSave+plotName, bbox_inches="tight")
        plt.close()
        with open(folderToSave+plotName.replace(".png", "distribution test.txt"), "w") as fh:
            fh.write(distributionResultsText)
        fh.close()
    else:
        print(distributionResultsText)
        plt.show()

if __name__ == "__main__":
    # compareLocalTopologyPrediction()
    plotPercentageCorrectTopologies(folderToSave="Results/DivAndTopoApplication/")
