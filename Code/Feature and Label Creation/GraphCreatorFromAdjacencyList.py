import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.decomposition
import sys
from sklearn.preprocessing import StandardScaler
from CellInSAMCenterDecider import CellInSAMCenterDecider

class GraphCreatorFromAdjacencyList:

    def __init__(self, adjacencyTableFilename, cellSizeFilename=None, sep=",",
                skipFooter=4, useEdgeWeight=False, invertEdgeWeight=False,
                useSharedWallWeight=False, useDistanceWeight=False):
        # ----- ATTENTION last 4 lines need to be skipped (cellSizeFilename), -----
        # ----- make sure this does not change with MorphoGraphX Version before applying script -----
        self.invertEdgeWeight = invertEdgeWeight
        self.useSharedWallWeight = useSharedWallWeight
        self.useDistanceWeight = useDistanceWeight
        self.skipFooter = skipFooter
        self.adjacencyTable = pd.read_csv(adjacencyTableFilename, sep=sep)
        self.adjacencyListDict = self.createAdjacencyListFromTableColumns()
        self.graph = nx.from_dict_of_lists(self.adjacencyListDict)
        if not cellSizeFilename is None:
            self.cellSizeFilename = cellSizeFilename
            self.cellSize = pd.read_csv(cellSizeFilename, sep=sep,
                                        skipfooter=self.skipFooter,
                                        index_col=0, engine="python")
            self.cellPosition = self.cellSize.iloc[:, 2:5].copy()
            self.cellSize = self.cellSize.iloc[:, 1].copy()
            if useEdgeWeight:
                self.propagateCellSizeToEdges(self.cellSize)
        else:
            self.cellSize = None

    def createAdjacencyListFromTableColumns(self, connectingFrom=0, connectingTo=1, sharedWallCol=2,
                                            ignoreSharedWallsOfZero=True):
        adjacencyListDict = {}
        fromNodes = self.adjacencyTable.iloc[:, connectingFrom]
        toNodes = self.adjacencyTable.iloc[:, connectingTo]
        sharedWall = self.adjacencyTable.iloc[:, sharedWallCol]
        for i in range(len(fromNodes)):
            currentNode1 = fromNodes[i]
            currentNode2 = toNodes[i]
            if ignoreSharedWallsOfZero and sharedWall[i] == 0:
                continue
            if not currentNode1 in adjacencyListDict.keys():
                adjacencyListDict[currentNode1] = []
            adjacencyListDict[currentNode1].append(currentNode2)
        return adjacencyListDict

    def propagateCellSizeToEdges(self, cellSize):
        missingLabelsInCellSizeTable = np.asarray(self.graph.nodes())[np.isin(np.asarray(self.graph.nodes()), cellSize.index.values, invert=True)]
        missingLabelsInAdjacencyTable = cellSize.index.values[np.isin(cellSize.index.values, np.asarray(self.graph.nodes()), invert=True)]
        if np.any(missingLabelsInCellSizeTable):
            print("The cell labels {} are missing in the cell size table and are removed from the graph.".format(missingLabelsInCellSizeTable))
            self.graph.remove_nodes_from(missingLabelsInCellSizeTable)
        if np.any(missingLabelsInAdjacencyTable):
            print("The cell labels {} are missing in the adjacency table.".format(missingLabelsInAdjacencyTable))
        nodesLabel = list(self.graph.nodes())
        self.graph = self.createWeightedGraph(cellSize, nodesLabel)
        nodeMapping = self.createNodeMapping(nodesLabel)
        nx.relabel_nodes(self.graph, nodeMapping, False)

    def createWeightedGraph(self, cellSize, nodesLabel):
        adjacencyMatrix = nx.to_numpy_matrix(self.graph)
        for j in range(len(nodesLabel)):
            currentNodeLabel = nodesLabel[j]
            currentCellSize = cellSize.loc[currentNodeLabel]
            for i in np.where(adjacencyMatrix[:, j]==1)[0]:
                adjacentNodeLabel = nodesLabel[i]
                adjacentCellSize = cellSize.loc[adjacentNodeLabel]
                if self.useSharedWallWeight:
                    sharedWall = self.determineSharedWall(currentNodeLabel, adjacentNodeLabel)
                    assert sharedWall != 0, "The nodes {} and {} have a sharedWall of 0, which is not allowed. Consider removing one or both nodes.".format(currentNodeLabel, adjacentNodeLabel)
                    weight = 1 / sharedWall
                elif self.useDistanceWeight:
                    distance = self.determineDistance(currentNodeLabel, adjacentNodeLabel)
                    assert distance != 0, "The nodes {} and {} have a distance of 0, which is not allowed. Consider removing one or both nodes.".format(currentNodeLabel, adjacentNodeLabel)
                    weight = 1 / distance
                else:
                    weight = (currentCellSize+adjacentCellSize) / 2
                if self.invertEdgeWeight:
                    weight = 1 / weight
                adjacencyMatrix[i, j] = weight
                adjacencyMatrix[j, i] = weight
        return nx.from_numpy_matrix(adjacencyMatrix)

    def determineDistance(self, currentNodeLabel, adjacentNodeLabel, dimensionMinus1=2):
        isCurrentNode = np.isin(self.cellPosition.index, currentNodeLabel)
        isAdjacentNode = np.isin(self.cellPosition.index, adjacentNodeLabel)
        currentNodePosition = self.cellPosition.iloc[isCurrentNode, :dimensionMinus1].to_numpy(copy=True)
        adjacentNodePosition = self.cellPosition.iloc[isAdjacentNode, :dimensionMinus1].to_numpy(copy=True)
        distance = np.sum(np.abs(currentNodePosition-adjacentNodePosition))
        return distance

    def determineSharedWall(self, currentNodeLabel, adjacentNodeLabel):
        isCurrentNode = np.asarray(self.adjacencyTable.iloc[:, 0] == currentNodeLabel)
        isAdjacentNode = np.asarray(self.adjacencyTable.iloc[:, 1] == adjacentNodeLabel)
        isCellWall = isCurrentNode & isAdjacentNode
        sharedCellWallArea = float(self.adjacencyTable.iloc[isCellWall,2])
        return sharedCellWallArea

    def createNodeMapping(self, nodesLabel):
        nodeMapping = {}
        for i in range(len(nodesLabel)):
            nodeMapping[i] = nodesLabel[i]
        return nodeMapping

    def PlotGraph(self, showEdgeWeights=False, colorLabels=False,
                centerCellLabels=None, centerRadius=30, highlightNodes=[],
                dividingCells=None, removeOutside=False, coloringProperty=None,
                withLabels=False):
        if self.cellSize is None:
            nx.draw(self.graph)
        else:
            self.edgeWidth = self.determineEdgeWidth(showEdgeWeights)
            self.nodePosition = self.extractNodePosition()
            self.nodeIdxLabels = np.asarray(list(self.graph.nodes))
            self.nodeColor = np.full(len(self.nodeIdxLabels), "#000000")# "#1f78b4")#blue color
            self.handleCentralCells(centerCellLabels, centerRadius, removeOutside)
            if len(highlightNodes) > 0:
                highlightNodes = np.asarray(highlightNodes)
                self.highlightNodesPosition  = np.where(np.isin(self.nodeIdxLabels, highlightNodes))[0]
                self.nodeColor[self.highlightNodesPosition] = "red"
            if not dividingCells is None:
                self.drawDividinAndNonDividingCells(dividingCells, coloringProperty)
            else:
                nx.draw(self.graph, width=1, pos=self.nodePosition, node_shape="o",
                        node_color=self.nodeColor, with_labels=withLabels, node_size=150)
        plt.show()

    def determineEdgeWidth(self, showEdgeWeights):
        if showEdgeWeights:
            edgeWidth = np.asarray(list(nx.get_edge_attributes(self.graph, 'weight').values()))
            edgeWidth /= np.max(edgeWidth)
            edgeWidth *= 3
        else:
            edgeWidth = None
        return edgeWidth

    def extractNodePosition(self):
        #pca, self.cellPosition = self.applyPCA(self.cellPosition, n_components=2)
        nodePosition = {}
        for i in list(self.cellPosition.index):
            nodePosition[i] = list(self.cellPosition.loc[i])[:2]
        return nodePosition

    def handleCentralCells(self, centerCellLabels, centerRadius, removeOutside):
        if not centerCellLabels is None:
            assert type(centerCellLabels) == list, "The given central cell/s {} need to be in a list".format(centerCellLabels)
            myCellInSAMCenterDecider = CellInSAMCenterDecider(self.cellSizeFilename,
                                        centerCellLabels, centerRadius=centerRadius)
            self.centralRegionLabels = np.asarray(myCellInSAMCenterDecider.GetCentralCells())
            self.highlightNodesPosition = np.isin(self.nodeIdxLabels, self.centralRegionLabels)
            if removeOutside:
                self.graph = self.graph.subgraph(self.centralRegionLabels)
            else:
                self.nodeColor[self.highlightNodesPosition] = "orange"

    def drawDividinAndNonDividingCells(self, dividingCells, coloringProperty):
        nx.draw(self.graph, width=self.edgeWidth, pos=self.nodePosition, node_shape="o",
                node_color=self.nodeColor, with_labels=False, node_size=0)#
        dividingCellLabels = []
        nonDividingCellLabels = []
        index = np.asarray(list(dividingCells.index))
        values = list(dividingCells.values)
        dividingNodeColor = []
        nonDividingNodeColor = []
        for cellLabel in self.centralRegionLabels:
            labeIdx = np.where(index == cellLabel)[0][0]
            if values[labeIdx][0] == 1:
                dividingNodeColor.append(coloringProperty.iloc[labeIdx])
                dividingCellLabels.append(cellLabel)
            else:
                nonDividingNodeColor.append(coloringProperty.iloc[labeIdx])
                nonDividingCellLabels.append(cellLabel)
        nx.draw(self.graph, width=self.edgeWidth, pos=self.nodePosition, node_shape="o",
                node_color=self.nodeColor, with_labels=False, node_size=0)
        nx.draw(self.graph, width=self.edgeWidth, pos=self.nodePosition, node_shape="o",
                node_color=dividingNodeColor, with_labels=False, node_size=150,
                nodelist=dividingCellLabels)
        nx.draw(self.graph, width=self.edgeWidth, pos=self.nodePosition, node_shape="v",
                node_color=nonDividingCellLabels, with_labels=False, node_size=150,
                nodelist=nonDividingCellLabels)

    def applyPCA(self, features, n_components=2):
        PCNames = ["principal component {}".format(i) for i in range(1, n_components+1)]
        normalisedFeatures = StandardScaler().fit_transform(features)
        pca = sklearn.decomposition.PCA(n_components=n_components)
        principalComponents = pca.fit_transform(normalisedFeatures)
        principalDf = pd.DataFrame(data=principalComponents,
                    columns=PCNames)
        principalDf.set_index(features.index, inplace=True)
        return pca, principalDf

    def AddCoordinatesPropertyToGraphFrom(self, geometryFilenameOrTable, sep=","):
        if type(geometryFilenameOrTable) == pd.core.frame.DataFrame:
            geometryData = geometryFilenameOrTable
        else:
            geometryData = pd.read_csv(geometryFilenameOrTable, sep=sep,
                                       skipfooter=self.skipFooter,
                                       engine="python")
        allCoordinates = geometryData[["Center_X", "Center_Y", "Center_Z"]].to_numpy()
        coordinates = {}
        for i in range(geometryData.shape[0]):
            coordinates[geometryData.iloc[i, 0]] = list(allCoordinates[i, :])
        nx.set_node_attributes(self.graph, coordinates, "coordinates")

    def GetGraph(self):
        return self.graph

    def PrintAdjacencyList(self):
        print(self.adjacencyListDict)

    def GetCellSizes(self):
        return self.cellSize

def main():
    from FeatureVectorCreator import FeatureVectorCreator
    from scipy import stats
    plant = "P2"#"P1"
    folder = "./Data/WT/{}/".format(plant)
    filename = "cellularConnectivityNetwork{}T0.csv".format(plant)
    cellSizeFilename = "area{}T0.csv".format(plant)
    from FeatureVectorCreator import FeatureVectorCreator
    from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
    from CellInSAMCenterDecider import CellInSAMCenterDecider
    from LabelCreator import LabelCreator
    connectivityNetworkFilename =  "cellularConnectivityNetwork{}T0.csv".format(plant)
    featuresToTest = ["clustering coefficient", "betweenness centrality", "page rank",
                        "eigenvector centrality", "katz centrality",
                        "current flow betweenness centrality",
                        "communicability betweenness centrality",
                        "load centrality", "harmonic centrality"]
    folderToSave = "Results/testingNetworkPropertiesWeightedVsUnweighted/"
    # columnToColor = 0
    centerCellLabels = [392] # None 38
    # invertEdgeWeight = False
    # useEdgeWeight = False
    # useSharedWallWeight = False
    # connectivityGraph = GraphCreatorFromAdjacencyList(folder + connectivityNetworkFilename).GetGraph()
    # myCellInSAMCenterDecider = CellInSAMCenterDecider(folder+cellSizeFilename, centerCellLabels, centerRadius=30)
    # centralRegionLabels = myCellInSAMCenterDecider.GetCentralCells()
    # featureMatrix = FeatureVectorCreator(connectivityGraph).GetFeatureMatrixDataFrame()
    # print(featureMatrix.columns)
    # positiveLabelsFilename = "parentLabeling{}T0T1.csv".format(plant)
    # myLabelCreator = LabelCreator(folder + positiveLabelsFilename, featureMatrix, allowedLabels=centralRegionLabels, areParametersFilenames=[True, False])
    # dividingCells = myLabelCreator.GetLabelVector()
    # weightedGraph = myGraphCreatorFromAdjacencyList.GetGraph()
    # coloringProperty = featureMatrix.iloc[:, columnToColor]
    # print(coloringProperty)
    invertEdgeWeight = False
    useEdgeWeight = True
    useSharedWallWeight = True
    coloringProperty = "orange"#None
    dividingCells = None
    showEdgeWeights = False
    removeOutside = False
    # centerCellLabels = [435, 597]
    myGraphCreatorFromAdjacencyList = GraphCreatorFromAdjacencyList(folder+filename,
                                                folder+cellSizeFilename,
                                                useEdgeWeight=useEdgeWeight,
                                                invertEdgeWeight=invertEdgeWeight,
                                                useSharedWallWeight=useSharedWallWeight)
    myGraphCreatorFromAdjacencyList.PlotGraph(centerCellLabels=None,
                                            showEdgeWeights=showEdgeWeights,
                                            dividingCells=dividingCells,
                                            removeOutside=removeOutside,
                                            coloringProperty=coloringProperty,
                                            highlightNodes=centerCellLabels,
                                            withLabels=False)

    # useEdgeWeight = False
    # myGraphCreatorFromAdjacencyList = GraphCreatorFromAdjacencyList(folder+filename,
    #                                             folder+cellSizeFilename,
    #                                             useEdgeWeight=useEdgeWeight,
    #                                             invertEdgeWeight=invertEdgeWeight,
    #                                             useSharedWallWeight=useSharedWallWeight)
    # unweightedGraph = myGraphCreatorFromAdjacencyList.GetGraph()
    # unweightedGraphProperty = nx.cluster.clustering(unweightedGraph)
    # weightedGraphProperty = nx.cluster.clustering(weightedGraph, weight="weight")
    # un = stats.zscore(list(unweightedGraphProperty.values()))
    # we = []
    # for i in unweightedGraphProperty.keys():
    #     we.append(weightedGraphProperty[i])
    # we = stats.zscore(we)
    # print(un[:4])
    # print(we[:4])

    # # selected features are the same, why? how to change it? , because weight or distance needs to be specified
    # unweightedFeatures = FeatureVectorCreator(unweightedGraph, useOwnFeatures=featuresToTest).GetFeatureMatrixDataFrame()
    # unweightedFeatures.to_csv(folderToSave+"unweightedGraph.csv")
    # weightedFeatures = FeatureVectorCreator(weightedGraph, useOwnFeatures=featuresToTest).GetFeatureMatrixDataFrame()
    # weightedFeatures.sort_index(inplace=True)
    # weightedFeatures.to_csv(folderToSave+"weightedGraph.csv")
    # # test if z-normalisation makes them similar again?
    # columns = unweightedFeatures.columns
    # index = unweightedFeatures.index
    # weightedFeatures = stats.zscore(weightedFeatures)
    # weightedFeatures = pd.DataFrame(weightedFeatures, columns=columns, index=index)
    # weightedFeatures.to_csv(folderToSave+"weightedGraphZNorm.csv")
    # unweightedFeatures = stats.zscore(unweightedFeatures)
    # unweightedFeatures = pd.DataFrame(unweightedFeatures, columns=columns, index=index)
    # unweightedFeatures.to_csv(folderToSave+"unweightedGraphZNorm.csv")

if __name__ == '__main__':
    main()
