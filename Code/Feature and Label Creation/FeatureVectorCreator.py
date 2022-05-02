import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sys
from scipy import stats

class FeatureVectorCreator (object):

    def __init__(self, graph, allowedLabels=None, additionalFeatures=None,
                zNormaliseFeaturesPerTissue=False, maxNormalise=False, secondNeighbourhoodProperties=True,
                useOwnFeatures=False, filename="", weightProperty="weight"):
        self.setProperties(graph, allowedLabels, additionalFeatures,
                    zNormaliseFeaturesPerTissue, maxNormalise, secondNeighbourhoodProperties,
                    useOwnFeatures, filename, weightProperty)
        self.setFeatureMatrixOfAllLabels()
        self.normaliseFeatureMatrix(zNormaliseFeaturesPerTissue, maxNormalise)
        self.featureMatrixDataFrame = self.converFeatureMatrixToDataFrame()

    def setProperties(self, graph, allowedLabels=None, additionalFeatures=None,
                zNormaliseFeaturesPerTissue=False, maxNormalise=False, secondNeighbourhoodProperties=True,
                useOwnFeatures=False, filename="", weightProperty="weight"):
        self.graph = graph
        self.percolationAttributeName = None
        self.allowedLabels = allowedLabels
        self.filename = filename
        self.allNodesOfGraph = list(self.graph.nodes)
        self.selectedNodes = list(self.graph.nodes)
        self.implementedFeatures = ["degree", "weighted node degree", "clustering coefficient", "degree centrality",
                                    "eccentricity","eccentricity normalised by radius",
                                    "absolute closeness centrality", "relative closeness centrality",
                                    "information centrality", "betweenness centrality",
                                    "log10 betweenness centrality",
                                    "eigenvector centrality", "page rank", "katz centrality",
                                    "current flow betweenness centrality",
                                    "communicability betweenness centrality",
                                    "load centrality", "harmonic centrality"]
        if not useOwnFeatures is False:
            self.features = useOwnFeatures.copy()
        else:
            self.features = ["degree", "weighted node degree", "clustering coefficient",
                            "information centrality", "betweenness centrality",
                            "eigenvector centrality", "page rank", "katz centrality",
                            "current flow betweenness centrality",
                            "communicability betweenness centrality",
                            "load centrality", "harmonic centrality"]
        # excluded features "eccentricity", "eccentricity normalised by radius", "relative closeness centrality",
        # add check to be sure all features are implemented
        self.featuresOnInducedGraph = ["size", "estrada index","abs graph density",
                                        "rel graph density", "avg path length",
                                        "algebraic connectivity"]
        # "assortativity", "assortativity relaxed","transitivity","clustering coefficient",
        if secondNeighbourhoodProperties:
            self.neighborhoodsOfInducedGraphs = [2]
        else:
            self.neighborhoodsOfInducedGraphs = []
        if additionalFeatures:
            try :
                self.features.extend(additionalFeatures)
            except:
                print("Enter your additional features in a vector.")
        self.isGraphWeighted = bool(nx.get_edge_attributes(self.graph, name=weightProperty))
        if self.isGraphWeighted:
            self.isGraphWeighted = weightProperty
        else:
            self.isGraphWeighted = None

    def setFeatureMatrixOfAllLabels(self):
        self.featureMatrix = self.calculateLocalAndLocalGlobalFeatures()
        if not self.allowedLabels is None:
            self.allowedLabels = np.asarray(self.allowedLabels).copy()
            isCellPresent = np.isin(self.allowedLabels, self.allNodesOfGraph)
            assert np.all(isCellPresent), "The features for the node/s {} should be calculated, but are not present in the graph {} with the nodes {}.".format(self.allowedLabels[np.invert(isCellPresent)], self.filename, self.allNodesOfGraph)
            self.removeNotAllowedLabels()
            self.orderByLabels()
        self.addFeaturesOffInducedGraphToFeatureMatrix(self.neighborhoodsOfInducedGraphs, self.featuresOnInducedGraph)
        return

    def calculateLocalAndLocalGlobalFeatures(self):
        featureMatrix = self.setupFeatureMatrix()
        for i in range(self.nrOfLocalAndLocalGlobalFeatures):
            if self.features[i] == "degree":
                featureMatrix[:, i] = self.getNodesDegrees()
            elif self.features[i] == "weighted node degree":
                featureMatrix[:, i] = self.calculateWeighstOfNodes()
            elif self.features[i] == "clustering coefficient":
                featureMatrix[:, i] = list(nx.cluster.clustering(self.graph, weight=self.isGraphWeighted).values())
            elif self.features[i] == "degree centrality":
                featureMatrix[:, i] = list(nx.degree_centrality(self.graph).values())
            elif self.features[i] == "eccentricity":
                featureMatrix[:, i] = list(nx.algorithms.distance_measures.eccentricity(self.graph).values())
            elif self.features[i] == "eccentricity normalised by radius":
                featureMatrix[:, i] = list(nx.algorithms.distance_measures.eccentricity(self.graph).values())
                radius = nx.radius(self.graph)
                featureMatrix[:, i] = np.asarray(featureMatrix[:, i])/radius
            elif self.features[i] == "absolute closeness centrality":
                featureMatrix[:, i] = list(nx.closeness_centrality(self.graph, distance=self.isGraphWeighted).values())/np.full(nrOfNodes, nrOfNodes-1)
            elif self.features[i] == "relative closeness centrality":
                featureMatrix[:, i] = list(nx.closeness_centrality(self.graph, distance=self.isGraphWeighted).values())
            elif self.features[i] == "information centrality":
                featureMatrix[:, i] = list(nx. information_centrality(self.graph, weight=self.isGraphWeighted).values())
            elif self.features[i] == "Stress centrality": # wrong implemented, calculates the same value as betweenness centrality
                featureMatrix[:, i] = self.calculateStressCentrality(self.graph)
            elif self.features[i] == "betweenness centrality":
                featureMatrix[:, i] = list(nx.betweenness_centrality(self.graph, weight=self.isGraphWeighted).values())
            elif self.features[i] == "log10 betweenness centrality":
                featureMatrix[:, i] = np.log10(list(nx.betweenness_centrality(self.graph, weight=self.isGraphWeighted).values()))
            elif self.features[i] == "eigenvector centrality":
                featureMatrix[:, i] = list(nx.eigenvector_centrality(self.graph, max_iter=10000, weight=self.isGraphWeighted).values())
            elif self.features[i] == "page rank":
                featureMatrix[:, i] = list(nx.pagerank(self.graph, alpha=0.85).values())
            elif self.features[i] == "vote rank":
                print("vote rank is not yet completely implemented")
                featureMatrix[:, i] = list(nx.voterank(self.graph).values())
                sys.exit()
            elif self.features[i] == "katz centrality":
                featureMatrix[:, i] = self.calcKatzCentrality(self.graph, igraphWeightProperty=self.isGraphWeighted)
            elif self.features[i] == "current flow betweenness centrality":
                featureMatrix[:, i] = list(nx.current_flow_betweenness_centrality(self.graph, weight=self.isGraphWeighted).values())
            elif self.features[i] == "communicability betweenness centrality":
                featureMatrix[:, i] = list(nx.communicability_betweenness_centrality(self.graph).values())
            elif self.features[i] == "load centrality":
                featureMatrix[:, i] = list(nx.load_centrality(self.graph, weight=self.isGraphWeighted).values())
            elif self.features[i] == "harmonic centrality":
                featureMatrix[:, i] = list(nx.harmonic_centrality(self.graph, distance=self.isGraphWeighted).values())
            elif self.features[i] == "percolation centrality":
                assert not self.percolationAttributeName is None, "The attribute where perculation is calculated on needs to be a value of the node, extractable with get_node_attributes."
                featureMatrix[:, i] = list(nx.percolation_centrality(self.graph, attribute=self.percolationAttributeName, weight=self.isGraphWeighted).values())
            else:
                print("You are wanting the feature {} which is not implemented. The list of implemented features is {}".format(self.features[i], self.implementedFeatures))
                sys.exit(1)
        return featureMatrix

    def setupFeatureMatrix(self):
        self.nrOfLocalAndLocalGlobalFeatures = len(self.features)
        self.nrOfFeaturesOnInducedGraph = len(self.featuresOnInducedGraph) * len(self.neighborhoodsOfInducedGraphs)
        self.nrOfAllFeatures = self.nrOfLocalAndLocalGlobalFeatures + self.nrOfFeaturesOnInducedGraph
        nrOfNodes = len(self.allNodesOfGraph)
        return np.zeros((nrOfNodes, self.nrOfAllFeatures))

    def getNodesDegrees(self):
        degrees = []
        for i in self.allNodesOfGraph:
            degrees.append(self.graph.degree(i))
        return degrees

    def calculateWeighstOfNodes(self):
        weightsOfNodes = []
        adjacencyList = nx.to_dict_of_dicts(self.graph)
        for node in self.allNodesOfGraph:
            currentWeight = 0
            for neighbour, edge in adjacencyList[node].items():
                currentWeight += edge.get("weight", 1)
            weightsOfNodes.append(currentWeight)
        return weightsOfNodes

    def calculateStressCentrality(self, G):
        betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
        for s in self.allNodesOfGraph:
            S, P, sigma = self._single_source_shortest_path_basic(G, s)
            betweenness = self._accumulate_basic(betweenness, S, P, sigma, s)
        betweenness = list(betweenness.values()) / np.full(1,2)
        return betweenness

    def _single_source_shortest_path_basic(self, G, s):
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)    # sigma[v]=0 for v in G
        D = {}
        sigma[s] = 1.0
        D[s] = 0
        Q = [s]
        while Q:   # use BFS to find shortest paths
            v = Q.pop(0)
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in G[v]:
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:   # this is a shortest path, count paths
                    sigma[w] += sigmav
                    P[w].append(v)  # predecessors
        return S, P, sigma

    def _accumulate_basic(self, betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w]
        return betweenness

    def calcKatzCentrality(self, graph, igraphWeightProperty=None):
        if igraphWeightProperty is None:
            katzCentrality = list(nx.katz_centrality(graph, max_iter=10000).values())
        else:
            highestEigenvalue = self.determineLargestEigenvalue(graph)
            # should work like that but it doesnt
            # katzCentrality = list(nx.katz_centrality(graph, alpha=1/highestEigenvalue,
            #                     weight=igraphWeightProperty, tol=1.0e-3).values())
            katzCentrality = list(nx.katz_centrality(graph, max_iter=10000).values())
        return katzCentrality

    def determineLargestEigenvalue(self, graph):
        laplacian = nx.normalized_laplacian_matrix(graph)
        eigenvalues = np.linalg.eigvals(laplacian.toarray())
        return np.max(eigenvalues)

    def removeNotAllowedLabels(self):
        isAllowedLabel = np.isin(self.allNodesOfGraph, self.allowedLabels)
        self.featureMatrix = self.featureMatrix[isAllowedLabel, :].copy()
        self.selectedNodes = np.asarray(self.allNodesOfGraph)[isAllowedLabel]

    def orderByLabels(self):
        order = np.zeros(len(self.allowedLabels), dtype=int)
        i = 0
        for currentAllowedLabel in self.allowedLabels:
            isNodeAllowed = self.selectedNodes == currentAllowedLabel
            assert np.sum(isNodeAllowed) <= 1, "The label {} should not exist more than once in the selected graph {} from {}".format(currentAllowedLabel, self.selectedNodes, self.filename)
            if np.sum(isNodeAllowed) == 1:
                order[i] = np.where(isNodeAllowed)[0]
                i += 1
        self.featureMatrix = self.featureMatrix[order, :]

    def addFeaturesOffInducedGraphToFeatureMatrix(self, radiusToInduceGaphOn, features):
        nrOfFeatures = len(features)
        for i in range(len(radiusToInduceGaphOn)):
            fromIdx = self.nrOfLocalAndLocalGlobalFeatures + (i * nrOfFeatures)
            toIdx = self.nrOfLocalAndLocalGlobalFeatures + (i * nrOfFeatures) + nrOfFeatures
            if self.allowedLabels is None:
                for j in range(len(self.allNodesOfGraph)):
                    inducedSubgraph = nx.ego_graph(self.graph, self.allNodesOfGraph[j], radiusToInduceGaphOn[i])
                    self.featureMatrix[j, fromIdx:toIdx] = self.calculateFeatoresOn(inducedSubgraph, features)
            else:
                for j in range(len(self.allowedLabels)):
                    inducedSubgraph = nx.ego_graph(self.graph, self.allowedLabels[j], radiusToInduceGaphOn[i])
                    self.featureMatrix[j, fromIdx:toIdx] = self.calculateFeatoresOn(inducedSubgraph, features)

    def calculateFeatoresOn(self, inducedSubgraph, allFeatures):
        featureVector = []
        for feature in allFeatures:
            if feature == "size":
                featureVector.append(inducedSubgraph.size())
            elif feature == "abs graph density":
                featureVector.append(self.calculateRelativeGraphDensity(inducedSubgraph) / (inducedSubgraph.number_of_nodes()-1))
            elif feature == "rel graph density":
                featureVector.append(self.calculateRelativeGraphDensity(inducedSubgraph))
            elif feature == "assortativity":
                featureVector.append(nx.degree_assortativity_coefficient(inducedSubgraph, weight=self.isGraphWeighted))
            elif feature == "assortativity relaxed":
                featureVector.append(nx.degree_assortativity_coefficient(inducedSubgraph, weight=self.isGraphWeighted))
            elif feature == "transitivity": # nochmal selbst rechnen
                featureVector.append(nx.transitivity(inducedSubgraph))
            elif feature == "estrada index":
                featureVector.append(nx.estrada_index(inducedSubgraph))
            elif feature == "avg path length":
                featureVector.append(nx.average_shortest_path_length(inducedSubgraph, weight=self.isGraphWeighted))
            elif feature == "algebraic connectivity":
                featureVector.append(nx.algebraic_connectivity(inducedSubgraph, weight=self.isGraphWeighted))
            elif feature == "clustering coefficient":
                featureVector.append(nx.average_clustering(inducedSubgraph, weight=self.isGraphWeighted))
            else:
                print("The feature {} is not yet implemented".format(feature))
        return featureVector

    def calculateRelativeGraphDensity(self, graph):
        return 2*graph.size()/graph.number_of_nodes()

    def normaliseFeatureMatrix(self, zNormaliseFeaturesPerTissue, maxNormalise):
        if zNormaliseFeaturesPerTissue is True:
            self.featureMatrix = stats.zscore(self.featureMatrix)
            if np.any(self.featureMatrix==np.NaN) or np.any(self.featureMatrix==np.inf):
                if np.any(self.featureMatrix==np.NaN):
                    print("An NaN was found at {}".format(np.where(self.featureMatrix==np.NaN)))
                if np.any(self.featureMatrix==np.inf):
                    print("An inf was found at {}".format(np.where(self.featureMatrix==np.inf)))
        elif maxNormalise is True:
            maxPerColumn = np.max(self.featureMatrix, axis=0)
            self.featureMatrix /= maxPerColumn

    def converFeatureMatrixToDataFrame(self):
        features = self.features
        for radius in self.neighborhoodsOfInducedGraphs:
            for additionalFeature in  self.featuresOnInducedGraph:
                features.append("{0} on {1} neighborhood".format(additionalFeature, radius))
        if self.allowedLabels is None:
            featureMatrixDataFrame = pd.DataFrame(self.featureMatrix, index=self.selectedNodes, columns=features)
        else:
            featureMatrixDataFrame = pd.DataFrame(self.featureMatrix, index=self.allowedLabels, columns=features)
        return featureMatrixDataFrame

    def GetFeatureMatrix(self):
        return self.featureMatrix

    def GetFeatureMatrixDataFrame(self):
        return self.featureMatrixDataFrame

    def SaveFeatureMatrixAsCSV(self, filename, sep=","):
        self.featureMatrixDataFrame.to_csv(filename, sep=sep)

def main():
    from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
    from CellInSAMCenterDecider import CellInSAMCenterDecider
    import sys
    # qnother feature could be degree distribution of degrees
    folder = "./Data/connectivityNetworks/"
    filename = ["cellularConnectivityNetworkP1T0.csv", "cellularConnectivityNetworkP1T1.csv", "cellularConnectivityNetworkP1T2.csv","cellularConnectivityNetworkP1T3.csv","cellularConnectivityNetworkP1T4.csv"]
    geometryFilenames = ["areaP1T0.csv", "areaP1T1.csv","areaP1T2.csv","areaP1T3.csv","areaP1T4.csv"]
    centerCellLabels = [[618, 467, 570], [5048, 5305], [5380, 5849, 5601], [6178, 6155, 6164], [6288, 6240]]
    for i in range(1,len(filename)):
        graph = GraphCreatorFromAdjacencyList(folder+filename[i], folder+geometryFilenames[i]).GetGraph()
        myCellInSAMCenterDecider = CellInSAMCenterDecider(folder+geometryFilenames[i], centerCellLabels[i], centerRadius=30)
        centralRegionLabels = myCellInSAMCenterDecider.GetCentralCells()
        myFeatureVectorCreation = FeatureVectorCreator(graph, allowedLabels=centralRegionLabels)
        filenameToSave = "./Data/connectivityNetworks/featureMatrixExampleGraph{}.csv".format(filename[i][-8:-4])
        print(myFeatureVectorCreation.GetFeatureMatrixDataFrame().shape)
        print(myFeatureVectorCreation.GetFeatureMatrixDataFrame().columns.values)
        #print(myFeatureVectorCreation.GetFeatureMatrix())
        sys.exit()
        #myFeatureVectorCreation.SaveFeatureMatrixAsCSV(filenameToSave)

if __name__ == '__main__':
    main()
