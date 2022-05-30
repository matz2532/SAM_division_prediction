import networkx as nx
import numpy as np
import pandas as pd
import sys
import time

sys.path.insert(0, "./Code/Feature and Label Creation/")

from DivAndTopoPredictor import DivAndTopoPredictor, findCentralNonPeripheralCells
from pathlib import Path
from PeripheralCellIdentifier import PeripheralCellIdentifier
from StandardTableFormater import StandardTableFormater
from TopologyPredictonDataCreator import TopologyPredictonDataCreator

class RandomisedTissuePrediction (DivAndTopoPredictor):
    confirmResultsManually = False

    def __init__(self, divisionProbability, topoPredProbability,
                 divSampleData, divSampleLabel,
                 runImprovedTopoPred=True, printOutProbablilites=True,
                 parentNetworks=None,
                 baseFolder="", plant="P2", timePoints=[0, 1, 2, 3], seed=42,
                 folderToSave="Results/DivAndTopoApplication/Random/Realistic/",
                 givenCellsNotToDivide=None,
                 divideAllCells=False,
                 correlateFeatures=True,
                 saveNetworks=True,
                 centralCellsList=None):
        self.divisionProbability = divisionProbability
        self.topoPredProbability = topoPredProbability
        self.divSampleData = divSampleData
        self.divSampleLabel = divSampleLabel
        self.runImprovedTopoPred = runImprovedTopoPred
        self.parentNetworks = parentNetworks
        self.baseFolder, self.plant, self.timePoints = baseFolder, plant, timePoints
        self.seed = seed
        self.folderToSave = folderToSave
        self.givenCellsNotToDivide = givenCellsNotToDivide
        self.divideAllCells = divideAllCells
        self.saveNetworks = saveNetworks
        self.centralCellsList = centralCellsList
        self.savePredictionsToFolder = None
        np.random.seed(self.seed)
        if printOutProbablilites:
            print("Probablilites of non-/dividing cell:", list(self.divisionProbability))
            print("Probablilites of class 0/1/2:", list(self.topoPredProbability))
        self.allNetworks = self.calcNetworks(self.baseFolder, self.plant, self.timePoints)
        self.estimateDividingCells()
        self.estimaeTopoChanges()
        if self.saveNetworks:
            self.simulateAllCellDivisions(folderToSave=self.folderToSave, networkBaseName="allPredNetworksDict_seed{}_{}_T{}.pkl".format(self.seed, "{}", "{}"))
        else:
            self.simulateAllCellDivisions()
        if correlateFeatures:
            self.estimateCorrelationsOfNonDivCells()

    def estimateDividingCells(self):
        self.dividingCellsOfTimePoint = []
        for t in self.timePoints:
            if not self.givenCellsNotToDivide is None:
                cellsToExclude = self.givenCellsNotToDivide[t]
            else:
                cellsToExclude = None
            predictedCellsToDivide = self.predictRandomCellDivisionOf(t, cellsToExclude=cellsToExclude)
            self.dividingCellsOfTimePoint.append(predictedCellsToDivide)

    def predictRandomCellDivisionOf(self, timePoint, cellsToExclude=None):
        currentNetwork = self.allNetworks[timePoint]
        nonPeripheralCells = PeripheralCellIdentifier(currentNetwork).GetNonPeripheralCells()
        if not cellsToExclude is None:
            isCellToKeep = np.isin(nonPeripheralCells, cellsToExclude, invert=True)
            nonPeripheralCells = nonPeripheralCells[isCellToKeep]
        if self.divideAllCells is True:
            return nonPeripheralCells
        else:
            nrOfCells = len(nonPeripheralCells)
            isDividing = np.random.choice(2, nrOfCells, p=self.divisionProbability)
            return nonPeripheralCells[isDividing==1]

    def estimaeTopoChanges(self):
        self.topoPairs, self.topoChangesOfTimePoint = [], []
        for t in self.timePoints:
            topoPairOfT = self.createTopoPairsToPredictOf(t)
            if self.runImprovedTopoPred:
                parentNetwork = self.parentNetworks[t]
                topoChangesOfT = self.predictImprovedRandomConnectionsFor(topoPairOfT, parentNetwork)
            else:
                topoChangesOfT = self.predictRandomConnectionsFor(topoPairOfT)
            self.topoPairs.append(topoPairOfT)
            self.topoChangesOfTimePoint.append(topoChangesOfT)

    def createTopoPairsToPredictOf(self, timePoint):
        currentNetwork = self.allNetworks[timePoint]
        dividingCells = self.dividingCellsOfTimePoint[timePoint]
        topoPairs = []
        for cell in dividingCells:
            for n in currentNetwork.neighbors(cell):
                topoPairs.append([cell, n])
        Formater = StandardTableFormater(currentPlantName=self.plant, currentTimePoint=timePoint)
        topoPairs = Formater.GetProperStandardFormatWithEveryEntryIn(topoPairs, extend=True)
        topoPairs = pd.DataFrame(topoPairs, columns=["plant", "time point", "dividing parent cell", "parent neighbor"])
        return topoPairs

    def predictImprovedRandomConnectionsFor(self, topoPairOfT, parentNetwork):
        dividingCells = np.unique(topoPairOfT.iloc[:, 2])
        connectionLabel = []
        for cell in dividingCells:
            idxDividingCell = np.where(topoPairOfT.iloc[:, 2] == cell)[0]
            neighbours = topoPairOfT.iloc[idxDividingCell, 3].to_numpy()
            labels = self.determineLabelsForNeighbours(neighbours, parentNetwork)
            connectionLabel.append(labels)
        return np.concatenate(connectionLabel)

    def determineLabelsForNeighbours(self, neighbours, parentNetwork):
        nrOfNeighbours = len(neighbours)
        startIdx = np.random.randint(0, nrOfNeighbours-1)
        startNeighbour = neighbours[startIdx]
        subgraph = parentNetwork.subgraph(neighbours)
        dfsNodes = nx.dfs_preorder_nodes(subgraph, source=startNeighbour)
        if nrOfNeighbours % 2 == 0:
            idxOfSecondClass2Neighbour = nrOfNeighbours // 2
        else:
            idxOfSecondClass2Neighbour = nrOfNeighbours // 2 + np.random.choice([0, 1])
        labels = np.full(nrOfNeighbours, 2)
        isFirstClass = np.random.choice([True, False])
        labels[1:idxOfSecondClass2Neighbour] = int(isFirstClass)
        labels[idxOfSecondClass2Neighbour+1:] = int(not isFirstClass)
        return labels

    def predictRandomConnectionsFor(self, topoPairOfT):
        nrOfTopoPairs = len(topoPairOfT)
        connectionLabel = np.random.choice(3, nrOfTopoPairs, p=self.topoPredProbability)
        return connectionLabel

    def estimateCorrelationsOfNonDivCells(self):
        predFeatures, actualFeatures = self.calcFeaturesForNonDivCellsInBoth()
        correlations = self.correlateFeatures(predFeatures, actualFeatures)
        np.save(f"{self.folderToSave}correlations_seed{self.seed}.npy", correlations)

def removePlantEntries(table, plantNameToRemove, plantNameColIdx=0, inverse=False):
    plantName = table.iloc[:, plantNameColIdx].to_numpy()
    if inverse:
        isSelectedPlant = np.isin(plantName, plantNameToRemove)
    else:
        isSelectedPlant = np.isin(plantName, plantNameToRemove, invert=True)
    table = table.iloc[isSelectedPlant, :]
    return table

def calcProbabilityOfLabel(divLabelTable, labelColIdx=-1):
    _, counts = np.unique(divLabelTable.iloc[:, labelColIdx], return_counts=True)
    return counts/divLabelTable.shape[0]

def printStatus(plantName, currentIteration, maxNrOfIterations, startTime, msgWith3FormatEntries="For {} {} of {} random tissue predictions are done. {} min"):
    timeAfterStart = np.round((time.time()-startTime)/60, 1)
    print(msgWith3FormatEntries.format(plantName, currentIteration, maxNrOfIterations, timeAfterStart))

def getParentNetworksOf(plantName=3, timeIdx=3, dataFolder="Data/WT/", centralCellsDict=None):
    if  centralCellsDict is None:
        centralCellsDict =  {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                            "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                            "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                            "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                            "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
    myTopologyPredictonDataCreator = TopologyPredictonDataCreator(dataFolder, 5, [plantName],centralCellsDict=centralCellsDict, skipFooterOfGeometryFile=4)
    data = myTopologyPredictonDataCreator.GetData()
    if type(timeIdx) == int:
        parentConnectivityNetwork = data[plantName]["graphCreators"][timeIdx].GetGraph()
    else:
        parentConnectivityNetwork = [data[plantName]["graphCreators"][t].GetGraph() for t in timeIdx]
    return parentConnectivityNetwork

def repeatedlyRunRandomisedTissuePrediction(givenSeedsToDo=None, nrOfRepetitions=None,
                                            testPlant="P2", timePoints=[0, 1, 2, 3],
                                            baseFolder = "./Data/WT/",
                                            verbose=1, printAfterIteration=5,
                                            runImprovedTopoPred=True, folderToSave="Results/DivAndTopoApplication/Random/Realistic/",
                                            givenCellsNotToDivide=None,
                                            divideAllCells=False,
                                            saveNetworks=False,
                                            mostCentralCells=[[392],  [553, 779, 527], [525], [1135]],
                                            divisionProbability=None, topoPredProbability=None):
    assert not givenSeedsToDo is None or not nrOfRepetitions is None, "Either givenSeedsToDo or nrOfRepetitions needs to be different than None."
    divLabelTable = pd.read_csv(baseFolder + "divEventData/manualCentres/allTopos/combinedLabels.csv")
    topoLabelTable = pd.read_csv(baseFolder + "topoPredData/diff/manualCentres/allTopos/combinedLabels.csv")
    divLabelTable = removePlantEntries(divLabelTable, testPlant)
    topoLabelTable = removePlantEntries(topoLabelTable, testPlant)
    if divisionProbability is None:
        divisionProbability = calcProbabilityOfLabel(divLabelTable)
    if topoPredProbability is None:
        topoPredProbability = calcProbabilityOfLabel(topoLabelTable)
    divSampleData = pd.read_csv(baseFolder + "divEventData/manualCentres/topology/combinedFeatures_topology_notnormalised.csv")
    divLabelTable = pd.read_csv(baseFolder + "divEventData/manualCentres/allTopos/combinedLabels.csv")
    parentNetworks = getParentNetworksOf(plantName=testPlant, timeIdx=timePoints, dataFolder=baseFolder, centralCellsDict={testPlant:mostCentralCells})
    startTime = time.time()
    if givenSeedsToDo is None:
        givenSeedsToDo = np.arange(nrOfRepetitions)
    else:
        nrOfRepetitions = len(givenSeedsToDo)
    if saveNetworks:
        centralCellsList = []
        for i, t in enumerate(timePoints):
            centralCells = findCentralNonPeripheralCells("{}{}/".format(baseFolder, testPlant), testPlant, t, mostCentralCells[i])
            centralCellsList.append(centralCells)
    else:
        centralCellsList = None
    for i, seed in enumerate(givenSeedsToDo):
        if verbose == 1:
            if seed % printAfterIteration == 0:
                printStatus(testPlant, i, nrOfRepetitions, startTime)
        elif verbose == 2:
            printStatus(testPlant, i, nrOfRepetitions, startTime)
        myRandomisedTissuePrediction = RandomisedTissuePrediction(divisionProbability, topoPredProbability,
                                            divSampleData.copy(), divLabelTable.copy(), parentNetworks=parentNetworks.copy(),
                                            baseFolder=baseFolder, plant=testPlant, seed=seed,
                                            printOutProbablilites=seed==0,
                                            runImprovedTopoPred=runImprovedTopoPred,
                                            folderToSave=folderToSave,
                                            givenCellsNotToDivide=givenCellsNotToDivide,
                                            divideAllCells=divideAllCells,
                                            saveNetworks=saveNetworks,
                                            centralCellsList=centralCellsList)

def calcMeanAndStdCorrelation(nrOfRepetitions, plantNames, folderToLoad="Results/DivAndTopoApplication/Random/Realistic/",
                         baseFilename="correlations_seed{}.npy", printOut=True):
    allCorrelations = []
    for plant in plantNames:
        for seed in range(nrOfRepetitions):
            cor = np.load(folderToLoad + plant + "/" + baseFilename.format(seed))
            allCorrelations.append(cor)
    meanCor = np.mean(allCorrelations, axis=0)
    stdCor = np.std(allCorrelations, axis=0)
    if printOut is True:
        print(meanCor)
        print(stdCor)
    return meanCor, stdCor

def wrapRandomTissuePredictionAndComparison(plantNames, mostCentralCellsDict, givenSeedsToDo, nrOfRepetitions, verbose=1,
        runImprovedTopoPred=False, divideAllCells=False, saveNetworks=True,
        baseFolder="./Data/WT/",
        folderToSave="Results/DivAndTopoApplication/Random/{}/",
        givenCellsNotToDivideFolder=None):
    if runImprovedTopoPred:
        folderToSave = folderToSave.format("Realistic", "{}")
    else:
        folderToSave = folderToSave.format("FullyRandom", "{}")
    if not givenCellsNotToDivideFolder is None:
        if divideAllCells is True:
            folderToSave += "fullyReversedPrediction/"
        else:
            folderToSave += "reversedPrediction/"
    elif not runImprovedTopoPred:
        folderToSave += "normal/"
    divLabelTable = pd.read_csv(baseFolder + "divEventData/manualCentres/allTopos/combinedLabels.csv")
    topoLabelTable = pd.read_csv(baseFolder + "topoPredData/diff/manualCentres/allTopos/combinedLabels.csv")
    divLabelTable = removePlantEntries(divLabelTable, plantNames)
    topoLabelTable = removePlantEntries(topoLabelTable, plantNames)
    if divisionProbability is None:
        divisionProbability = calcProbabilityOfLabel(divLabelTable)
    if topoPredProbability is None:
        topoPredProbability = calcProbabilityOfLabel(topoLabelTable)
    for plant in plantNames:
        folderToSavePlantResults = folderToSave + plant + "/"
        Path(folderToSavePlantResults).mkdir(parents=True, exist_ok=True)
        mostCentralCells = mostCentralCellsDict[plant]
        if not givenCellsNotToDivideFolder is None:
            givenCellsNotToDivide = np.load(givenCellsNotToDivideFolder+plant+"/dividingCellsOfTimePoint.pkl", allow_pickle=True)
        repeatedlyRunRandomisedTissuePrediction(givenSeedsToDo=givenSeedsToDo, nrOfRepetitions=nrOfRepetitions,
                                                testPlant=plant, baseFolder=baseFolder,
                                                runImprovedTopoPred=runImprovedTopoPred, folderToSave=folderToSavePlantResults,
                                                verbose=verbose, givenCellsNotToDivide=givenCellsNotToDivide,
                                                divideAllCells=divideAllCells, saveNetworks=saveNetworks,
                                                mostCentralCells=mostCentralCells,
                                                divisionProbability=divisionProbability,
                                                topoPredProbability=topoPredProbability)
    meanCor, stdCor = calcMeanAndStdCorrelation(nrOfRepetitions, plantNames=plantNames, folderToLoad=folderToSave)
    np.save(folderToSave+"meanCorrelations.npy", meanCor)
    np.save(folderToSave+"stdCorrelations.npy", stdCor)
    # cor = np.load("Results/DivAndTopoApplication/correlations.npy")#
    # diff = cor - meanCor
    # argsort = np.argsort(cor)[::-1]
    # print(list(cor[argsort]))
    # print(list(meanCor[argsort]))
    # print("diff", diff[argsort])

def mainRandomiseWithMultiplePlants():
    plantNames = ["P2", "P9"]
    mostCentralCellsDict = {"P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P9":[[1047, 721, 1048], [7303, 7533], [6735, 7129], [2160, 2228], [7366, 7236]]}
    nrOfRepetitions = 100
    givenSeedsToDo = np.arange(nrOfRepetitions)
    givenCellsNotToDivideFolder = "Results/DivAndTopoApplication/"
    wrapRandomTissuePredictionAndComparison(plantNames=plantNames,
                                            mostCentralCellsDict=mostCentralCellsDict,
                                            givenSeedsToDo=givenSeedsToDo,
                                            nrOfRepetitions=nrOfRepetitions,
                                            runImprovedTopoPred=False,
                                            divideAllCells=True,
                                            givenCellsNotToDivideFolder=givenCellsNotToDivideFolder)
    # timePoints=[0, 1, 2, 3]
    # testPlant = "P2"
    # baseFolder = "./Data/WT/"
    # divLabelTable = pd.read_csv(baseFolder + "divEventData/allTopos/combinedLabels.csv")
    # topoLabelTable = pd.read_csv(baseFolder + "topoPredData/diff/manualCentres/allTopos/combinedLabels.csv")
    # divLabelTable = removePlantEntries(divLabelTable, testPlant)
    # topoLabelTable = removePlantEntries(topoLabelTable, testPlant)
    # divisionProbability = calcProbabilityOfLabel(divLabelTable)
    # topoPredProbability = calcProbabilityOfLabel(topoLabelTable)
    # divSampleData = pd.read_csv(baseFolder + "divEventData/topology/combinedFeatures_topology_notnormalised.csv")
    # divLabelTable = pd.read_csv(baseFolder + "divEventData/allTopos/combinedLabels.csv")
    # parentNetworks = getParentNetworksOf(plant=testPlant, timeIdx=timePoints, dataFolder=baseFolder)
    # myRandomisedTissuePrediction = RandomisedTissuePrediction(divisionProbability, topoPredProbability,
    #                                     divSampleData, divLabelTable, parentNetworks=parentNetworks,
    #                                     baseFolder=baseFolder, plant=testPlant)

if __name__ == '__main__':
    mainRandomiseWithMultiplePlants()
