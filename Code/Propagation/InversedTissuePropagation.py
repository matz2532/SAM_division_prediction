import numpy as np

class InversedTissuePropagation (DivAndTopoPredictor):

    def __init__(self, divPredModel, topoPredModel, divSampleData, baseFolder,
                 plant, timePoints, divSampleLabel, seed=42,
                 confirmResultsManually=False, useBioFeatures=True):
        self.divPredModel = divPredModel
        self.topoPredModel = topoPredModel
        self.divSampleData = divSampleData
        self.baseFolder = baseFolder
        self.plant = plant
        self.timePoints = timePoints
        self.divSampleLabel = divSampleLabel
        self.confirmResultsManually = confirmResultsManually
        self.useBioFeatures = useBioFeatures
        np.random.seed(seed)
        self.excludedCells = {}
        self.topoPairs = {}
        self.setup()
        # 'dividingCellArea', 'neighbourCellArea', 'sharedWall', 'distance', 'dividingCellPerimeter', 'neighbourCellPerimeter'
        self.bioMeanPar = [43.40099969924809, 35.96200546365913, 4.484214656641601, 7.030076776415971, 18670.643107769425, 16883.23909774436]
        self.bioStdPar = [6.321010865033345, 8.60354563627134, 1.989491519293842, 1.2846260455372815, 14662.624454499268, 13522.844964626294]
        self.predDivEvents()
        self.topSampleData = self.calcTopoDataOfDividingCells()
        self.predTopoChanges()
        if self.confirmResultsManually:
            self.printTopoPredLabelsCount()
        self.simulateAllCellDivisions()
        self.estimateCorrelationsOfNonDivCells()

    def setup(self):
        self.preprocessDivSampleData()
        self.zNormTestDataFromTrainPar(self.plant)
        self.bioMeanPar, self.bioStdPar = self.calcBioFeatureMeanStdToNormalise(self.plant)
        self.allNetworks = self.calcNetworks(self.baseFolder, self.plant, self.timePoints)

    def preprocessDivSampleData(self):
        duplicateRemover = DuplicateDoubleRemover(self.divSampleData)
        self.duplicateColIdx = duplicateRemover.GetDuplicateColIdx() - 3 - self.useBioFeatures
        self.divSampleData = duplicateRemover.GetDoublesReducedTable()

    def zNormTestDataFromTrainPar(self, testPlant="P2"):
        isTestPlant = self.divSampleData.iloc[:, 0] == testPlant
        idxOfTrainPlants = np.where(np.invert(isTestPlant))[0]
        idxOfTestPlant = np.where(isTestPlant)[0]
        self.divSampleData.iloc[idxOfTrainPlants, 3:], self.meanStdPar = self.doZNormalise(self.divSampleData.iloc[idxOfTrainPlants, 3:], returnParameter=True)
        self.divSampleData.iloc[idxOfTestPlant, 3:] = self.doZNormalise(self.divSampleData.iloc[idxOfTestPlant, 3:], useParameters=self.meanStdPar)
        # as area is the first feature given but not the first when later used for network feature normalisation
        if self.useBioFeatures:
            self.meanStdPar = [self.meanStdPar[0][1:], self.meanStdPar[1][1:]]

    def calcBioFeatureMeanStdToNormalise(self, testPlant, printOut=True):
        topoSampleDataFilename = "Data/WT/topoPredData/diff/manualCentres/bio/combinedFeatures_bio_notnormalised.csv"
        bioTopoSampleData = pd.read_csv(topoSampleDataFilename)
        isTrainValPlant = np.invert(bioTopoSampleData.iloc[:, 0]==testPlant)
        idxOfTissue = np.where(isTrainValPlant)[0]
        bioTrainVal = bioTopoSampleData.iloc[idxOfTissue, 4:]
        _, bioMeanStdPar = self.doZNormalise(bioTrainVal, returnParameter=True)
        if printOut:
            print(list(bioMeanStdPar[0]), list(bioMeanStdPar[1]))
        return bioMeanStdPar

    def doZNormalise(self, X_train, useParameters=None, returnParameter=False):
        if not useParameters is None:
            mean = useParameters[0]
            std = useParameters[1]
        else:
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
        X_train = (X_train-mean)/std
        if returnParameter:
            return [X_train, [mean, std]]
        else:
            return X_train

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

    def calcTopoDataOfDividingCells(self):
        featuresOfTopoPred = []
        for i, dividingCells in enumerate(self.dividingCellsOfTimePoint):
            topoFeature = self.createTopoDataFor(dividingCells, self.timePoints[i])
            featuresOfTopoPred.append(topoFeature)
        return featuresOfTopoPred

    def createTopoDataFor(self, dividingCells, timePoint, addBioFeatures=True):
        self.excludedCells[timePoint] = []
        divCellNeighbourPairs = self.extractTopoPredPairs(dividingCells, timePoint)
        topoFeatures = self.calcTopoFeaturesFor(divCellNeighbourPairs, timePoint)
        if addBioFeatures:
            bioFeatureTable = self.calcBioFeatures(divCellNeighbourPairs, timePoint)
            topoFeatures = np.concatenate([bioFeatureTable.iloc[:, 4:].to_numpy(), topoFeatures], axis=1)
            self.topoPairs[timePoint] = bioFeatureTable.iloc[:, :4]
        return topoFeatures

    def extractTopoPredPairs(self, dividingCells, timePoint):
        idxOfTissue = self.selectIdxOf(self.divSampleData, self.plant, timePoint)
        fullNetwork = self.allNetworks[timePoint]
        divCellNeighbourPairs = []
        cellsOfTissue = self.selectDivPredFeatures(timePoint, startCol=0).iloc[:, 2].to_numpy()
        for d in dividingCells:
            neighbors = list(fullNetwork.neighbors(d))
            addPairs = self.doAllNeighborsExistInFeatures(neighbors, cellsOfTissue)
            if not addPairs:
                addPairs = self.canMissingNeighborsFeaturesBeCalculated(neighbors, cellsOfTissue, fullNetwork)
            if addPairs:
                for n in neighbors:
                    divCellNeighbourPairs.append([d, n])
            else:
                self.excludedCells[timePoint].append(d)
        return np.asarray(divCellNeighbourPairs)

    def doAllNeighborsExistInFeatures(self, neighborCells, cellsOfTissue):
        return len(neighborCells) == np.sum(np.isin(neighborCells, cellsOfTissue))

    def canMissingNeighborsFeaturesBeCalculated(self, neighborCells, cellsOfTissue, fullNetwork):
        allNeighborsNeighbor = []
        for n in neighborCells:
            neighborsNeighbor = list(fullNetwork.neighbors(n))
            allNeighborsNeighbor.extend(neighborsNeighbor)
        allNeighborsNeighbor = np.unique(allNeighborsNeighbor)
        peripheralCellIdentifier = PeripheralCellIdentifier(fullNetwork, testNetwork=False)
        peripheralNeighborsNeighbor = peripheralCellIdentifier.findPeripheralCells(allNeighborsNeighbor)
        return len(peripheralNeighborsNeighbor) == 0

    def calcTopoFeaturesFor(self, divCellNeighbourPairs, timePoint, networkFeatureStartCol=4,
                            concatDividingCellsFeatures=True):
        allTopoFeatures = []
        featuresOfTissue = self.selectDivPredFeatures(timePoint, startCol=0)
        cells = featuresOfTissue.iloc[:, 2].to_numpy()
        isNeighborFeatureMissing = np.isin(divCellNeighbourPairs[:, 1], cells, invert=True)
        missingCells = np.unique(divCellNeighbourPairs[isNeighborFeatureMissing, 1])
        missingCellsFeatures = self.calcMissingCellsFeatures(missingCells, timePoint)
        np.save("Results/DivAndTopoApplication/missingCellsFeatures.npy", missingCellsFeatures)
        missingCellsFeatures = np.load("Results/DivAndTopoApplication/missingCellsFeatures.npy")
        cells = np.concatenate([cells, missingCells])
        featuresOfTissue = featuresOfTissue.iloc[:, networkFeatureStartCol:].to_numpy()
        featuresOfTissue = np.concatenate([featuresOfTissue, missingCellsFeatures], axis=0)
        nrOfRows = featuresOfTissue.shape[1]
        rowToSplit = nrOfRows//2
        for d, n in divCellNeighbourPairs:
            dIdx = np.where(cells==d)[0]
            nIdx = np.where(cells==n)[0]
            diffFeatures = featuresOfTissue[nIdx, :] - featuresOfTissue[dIdx, :]
            if concatDividingCellsFeatures:
                neighbourCellFeatures = featuresOfTissue[nIdx, :]
                currentFeatures = np.concatenate([diffFeatures, neighbourCellFeatures], axis=1)
            else:
                currentFeatures = diffFeatures
            allTopoFeatures.append(currentFeatures)
        allTopoFeatures = np.concatenate(allTopoFeatures, axis=0)
        return allTopoFeatures

    def calcMissingCellsFeatures(self, missingCells, timePoint):
        cellSizeFilename = "./Data/WT/{}/area{}T{}.csv".format(self.plant, self.plant, timePoint)
        graphFilename = "./Data/WT/{}/cellularConnectivityNetwork{}T{}.csv".format(self.plant, self.plant, timePoint)
        weightingProperties = {"unweighted": [False, False, False, False],
                                "weighted by area": [True, True, False, False],
                                "weighted by wall":[True, False, True, False],
                                "weighted by distance":[True, False, False, False]}
        allFeatures = []
        for name, weightProperty in weightingProperties.items():
            useEdgeWeight, invertEdgeWeight, useSharedWallWeight, useDistanceWeight = weightProperty
            graphCreator = GraphCreatorFromAdjacencyList(graphFilename,
                                            cellSizeFilename=cellSizeFilename,
                                            useEdgeWeight=useEdgeWeight,
                                            invertEdgeWeight=invertEdgeWeight,
                                            useSharedWallWeight=useSharedWallWeight,
                                            useDistanceWeight=useDistanceWeight)
            if useDistanceWeight:
                graphCreator.AddCoordinatesPropertyToGraphFrom(cellSizeFilename)
            currentGraph = graphCreator.GetGraph()
            featureCreator = FeatureVectorCreator(currentGraph, allowedLabels=missingCells,
                            filename=graphFilename)
            # featureCreator.SaveFeatureMatrixAsCSV("Temporary/recreatedFeatures{}.csv".format(name))
            featureVector = featureCreator.GetFeatureMatrix()
            allFeatures.append(featureVector)
        allFeatures = np.concatenate(allFeatures, axis=1)
        allFeatures = np.delete(allFeatures, self.duplicateColIdx, axis=1)
        meanStdPar = [self.meanStdPar[0].to_numpy(), self.meanStdPar[1].to_numpy()]
        allFeatures = self.doZNormalise(allFeatures, useParameters=meanStdPar)
        return allFeatures

    def calcBioFeatures(self, divCellNeighbourPairs, timePoint):
        nrOfCells = divCellNeighbourPairs.shape[0]
        plantNames = np.full((nrOfCells, 1), self.plant)
        timePoints = np.full((nrOfCells, 1), timePoint)
        topoStandardTable = np.concatenate([plantNames, timePoints, divCellNeighbourPairs], axis=1)
        topoStandardTable = pd.DataFrame(topoStandardTable, columns=["plant", "time point", "dividing parent cell", "parent neighbor"])
        topoStandardTable = topoStandardTable.astype({"time point":int, "dividing parent cell":int, "parent neighbor":int})
        bioFeatures = BiologicalFeatureCreatorForNetworkRecreation(baseFolder=self.baseFolder, oldFeatureTable=topoStandardTable).CreateBiologicalFeatures()
        bioMeanPar = [self.bioMeanPar, self.bioStdPar]
        bioFeatures.iloc[:, 4:] = self.doZNormalise(bioFeatures.iloc[:, 4:], useParameters=bioMeanPar)
        return bioFeatures

    def predTopoChanges(self):
        self.topoChangesOfTimePoint = []
        for t in self.timePoints:
            topoFeatures = self.topSampleData[t]
            topoChanges = self.topoPredModel.predict(topoFeatures)
            self.topoChangesOfTimePoint.append(topoChanges)

    def simulateAllCellDivisions(self):
        self.tPlusOneNetworks = []
        for t in self.timePoints:
            nextTimesNetwork = self.applyDivAndTopoPred(t, self.confirmResultsManually)
            self.tPlusOneNetworks.append(nextTimesNetwork)

    def applyDivAndTopoPred(self, timePoint, plotAndPrintResults=False):
        oldNetwork = self.allNetworks[timePoint]
        dividingCells = self.dividingCellsOfTimePoint[timePoint]
        topoPredPairs = self.combineTopoPredAndPairInfos(timePoint)
        orderOfDivCells = self.determineOrderOfInmplementingChanges(topoPredPairs)
        newNetwork = self.propagateNetwork(oldNetwork, orderOfDivCells, topoPredPairs)
        if plotAndPrintResults:
            self.plotAndPrintDivTopoApplication(timePoint, newNetwork, orderOfDivCells)
        return newNetwork

    def combineTopoPredAndPairInfos(self, timePoint):
        topologicalChanges = self.topoChangesOfTimePoint[timePoint]
        topoPairs = self.topoPairs[timePoint]
        topoPredPairs = pd.concat([topoPairs, pd.DataFrame(topologicalChanges, columns=["labels"])], axis=1)
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
                print("The selected dividing cell {} is not inside the topology prediction olumn dividing cells {}".format(d, np.unique(topoPredPairs.iloc[:, 2])))
        return np.asarray(nrOfDivNeighbors), np.asarray(labelsOfDivNeighbors)

    def getIdxOf(self, entry, table, colIdx=2):
        isEntry = np.isin(table.iloc[:, colIdx], entry)
        return np.where(isEntry)[0]

    def propagateNetwork(self, oldNetwork, orderOfDivCells, topoPredPairs):
        newNetwork = oldNetwork.copy()
        nodeIds = list(newNetwork.nodes())
        newCellId = np.max(nodeIds) + 1
        for i in range(len(orderOfDivCells)):
            divCell = orderOfDivCells[i]
            self.removeDivNeighborEntries(divCell, topoPredPairs)
            divNeighborEdgesToRemove = np.asarray(list(oldNetwork.neighbors(divCell)))
            divNeighborEdgesToRemove = divNeighborEdgesToRemove[np.isin(divNeighborEdgesToRemove, orderOfDivCells[:i], invert=True)]
            for n in divNeighborEdgesToRemove:
                newNetwork.remove_edge(divCell, n)
            remainingNeighbors = list(newNetwork.neighbors(divCell))
            newNetwork.add_node(newCellId) # divCell is always cell A and newCellId always cell B
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
            newCellId += 1
        return newNetwork

    def removeDivNeighborEntries(self, divCell, topoPredPairs):
        self.idxOfDivNeighbors = self.getIdxOf(divCell, topoPredPairs, colIdx=3)
        if len(self.idxOfDivNeighbors) > 0:
            topoPredPairs.drop(topoPredPairs.index.to_numpy()[self.idxOfDivNeighbors], axis=0, inplace=True)

    def determinePossibleProblems(self, topoPredPairs):
        divCells = np.unique(topoPredPairs.iloc[:, 2])
        idxOfDivNeighbors = self.getIdxOf(divCells, topoPredPairs, colIdx=3)

    def determineNeighborLabelDict(self, divCell, topoPredPairs):
        idxOfDivCell = self.getIdxOf(divCell, topoPredPairs)
        neighbors = topoPredPairs.iloc[idxOfDivCell, 3]
        labels = topoPredPairs.iloc[idxOfDivCell, 4]
        return dict(zip(neighbors, labels))

    def estimateCorrelationsOfNonDivCells(self):
        predFeatures, actualFeatures = self.calcFeaturesForNonDivCellsInBoth()
        np.save("Results/DivAndTopoApplication/actualFeatures.npy", actualFeatures)
        np.save("Results/DivAndTopoApplication/predFeatures.npy", predFeatures)
        correlations = self.correlateFeatures(predFeatures, actualFeatures)
        print(correlations)
        np.save("Results/DivAndTopoApplication/correlations.npy", correlations)

    def calcFeaturesForNonDivCellsInBoth(self):
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
        # pickle.dump(mappedCells, open("Results/DivAndTopoApplication/mappedCells{}T{}.pkl".format(self.plant, timePoint), "wb"))
        # mappedCells = pickle.load(open("Results/DivAndTopoApplication/mappedCells{}T{}.pkl".format(self.plant, timePoint), "rb"))
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
        # allFeatures = self.doZNormalise(allFeatures, useParameters=meanStdPar)
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

def main():
    from DivAndTopoPredictor import loadTestModelsAndData
    featureSet = "topoAndBio"# "allTopos"
    baseFolder = "./Data/WT/"
    plant = "P2"
    timePoints = [0,1,2,3]
    divPredModel, topoPredModel, divSampleData, divSampleLabel = loadTestModelsAndData(baseFolder, featureSet)
    myInversedTissuePropagation = InversedTissuePropagation(divPredModel, topoPredModel,
                                                divSampleData, baseFolder,
                                                plant, timePoints,
                                                divSampleLabel=divSampleLabel)

if __name__ == '__main__':
    main()
