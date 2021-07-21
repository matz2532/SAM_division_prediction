import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "./Code/")

from utils import FooterExtractor

class PeripheralCellIdentifier (object):

    def __init__(self, network, tissueName=None, testNetwork=True,
                 logFilename="PeripheralCellIdentifierLog_{}.txt", folderToSaveLog=""):
        self.network = network
        self.tissueName = tissueName
        self.folderToSaveLog = folderToSaveLog
        todayDate = datetime.date.today()
        if "{}" in logFilename:
            self.logFilename = self.folderToSaveLog + logFilename.format(todayDate)
        else:
            self.logFilename = self.folderToSaveLog + "PeripheralCellIdentifierLog.txt"
        if testNetwork:
            nodesToTest = self.network.nodes()
            self.peripheralCells = self.findPeripheralCells(nodesToTest)
            self.nonPeripheralCells = self.calcNonPeripheralCells(self.peripheralCells, self.network)
        else:
            self.peripheralCells = None
            self.nonPeripheralCells = None

    def findPeripheralCells(self, nodesToTest):
        peripheralCells = []
        for node in nodesToTest:
            if self.isPeripheralCell(node):
                peripheralCells.append(node)
        return np.asarray(peripheralCells)

    def isPeripheralCell(self, node):
        self.currentNode = node
        neighbours = self.network.neighbors(node)
        subgraph = self.network.subgraph(neighbours).copy()
        subgraph = self.removeTriangleFromSubgraph(subgraph)
        doNeighborsFormCycle = self.neighborsFormCycle(subgraph)
        return not doNeighborsFormCycle

    def removeTriangleFromSubgraph(self, subgraph):
        nrOfTrianglesPerCell = nx.triangles(subgraph)
        nrOfTriangles = np.asarray(list(nrOfTrianglesPerCell.values()))
        cellHasTriangle = nrOfTriangles > 0
        while np.any(cellHasTriangle):
            self.removeEdgeOfTriangle(subgraph, cellHasTriangle)
            nrOfTrianglesPerCell = nx.triangles(subgraph)
            nrOfTriangles = np.asarray(list(nrOfTrianglesPerCell.values()))
            cellHasTriangle = nrOfTriangles > 0
        return subgraph

    def removeEdgeOfTriangle(self, subgraph, cellHasTriangle):
        nodes = np.asarray(list(subgraph.nodes()))
        nodesMakingUpTriangles = nodes[cellHasTriangle]
        if len(nodesMakingUpTriangles) == 3:
            degree = np.asarray(nx.degree(subgraph, nodesMakingUpTriangles))
            argsort = np.argsort(degree[:, 1])
            argsort = argsort[::-1]
            nodesMakingUpTriangles = nodesMakingUpTriangles[argsort]
            u = nodesMakingUpTriangles[0]
            v = nodesMakingUpTriangles[1]
        else:
            u, v = self.findNodesMakingFirstTriangle(subgraph, nodesMakingUpTriangles)
        subgraph.remove_edge(u, v)
        if not self.tissueName is None:
            textToLog = "tissue {} removed edge ({}, {}) around {}\n".format(self.tissueName, u, v, self.currentNode)
            file = open(self.logFilename, "a")
            file.write(textToLog)
            file.close()

    def findNodesMakingFirstTriangle(subgraph, nodesMakingUpTriangles):
        print("More than 3 cells make up triangles: {}".format(nodesMakingUpTriangles))
        print("not yet implemented!")
        nx.draw_spectral(subgraph)
        plt.show()
        sys.exit()

    def neighborsFormCycle(self, graph):
        try:
            nx.find_cycle(graph)
            isCyclic = True
        except:
            isCyclic = False
        return isCyclic

    def calcNonPeripheralCells(self, peripheralCells, network):
        allCells = np.asarray(network.nodes())
        nonPeripheralCells = allCells[np.isin(allCells, peripheralCells, invert=True)]
        return nonPeripheralCells

    def GetPeripheralCells(self):
        return self.peripheralCells

    def GetNonPeripheralCells(self):
        return self.nonPeripheralCells

    def ValidateWithKnownCells(self, knownPeripheralCells, isFilename=True, showPlot=True):
        if isFilename:
            table = pd.read_csv(knownPeripheralCells)
            isPeripheralCell = table["Value"] == 1
            cellLabel = table["Label"]
            knownPeripheralCells = np.asarray(cellLabel[isPeripheralCell])
            knownPeripheralCells = knownPeripheralCells.astype(int)
        isObservedInExcpected = np.isin(knownPeripheralCells, self.peripheralCells)
        isExcpectedInObserved = np.isin(self.peripheralCells, knownPeripheralCells)
        print("nr of observed in expected {}/{}".format(np.sum(isObservedInExcpected), len(knownPeripheralCells)))
        print("nr of expected in observed {}/{}".format(np.sum(isExcpectedInObserved), len(self.peripheralCells)))
        nodes = np.asarray(list(self.network.nodes()))
        print("total nr of nodes", len(nodes))
        nodeColor = np.full(len(nodes), "   black")
        isObservedNode = np.isin(nodes, self.peripheralCells)
        isExpectedNode = np.isin(nodes, knownPeripheralCells)
        print("nr of observed peripheralCells being in nodes", np.sum(np.isin(self.peripheralCells, nodes)), len(self.peripheralCells))
        print("nr of expected peripheralCells being in nodes", np.sum(np.isin(knownPeripheralCells, nodes)), len(knownPeripheralCells))
        isObservedAndExpected = isExpectedNode & isObservedNode
        nodeColor[isObservedNode] = "yellow"
        nodeColor[isExpectedNode] = "blue"
        nodeColor[isObservedAndExpected] = "green"
        nodeColor[nodeColor== "   black"] = "gray"
        nx.draw_spectral(self.network, node_color=nodeColor, with_labels=True)
        if showPlot:
            plt.show()

    def ValidateCreatingHeatmap(self, geometryFilename, saveToFilename, sep=",", skippedFooter=4):
        table = pd.read_csv(geometryFilename, sep=sep, skipfooter=skippedFooter, engine="python")
        valueOfIsPeripheral = np.zeros(len(table["Value"]))
        cellLabel = table["Label"]
        isCellPeripheral = np.isin(cellLabel, self.peripheralCells)
        valueOfIsPeripheral[isCellPeripheral] = 1
        table["Value"] = valueOfIsPeripheral
        table.to_csv(saveToFilename, sep=sep, index=False)
        FooterExtractor(fileToOpen=geometryFilename, extractFooter=skippedFooter,
                        saveToFilename=saveToFilename)

def main():
    from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
    from pathlib import Path
    plant = "P2"
    timePoint = "T0"
    showPlot = True
    dataFolder = "./Data/WT/"
    folder = "{}{}/".format(dataFolder, plant)
    filename = "cellularConnectivityNetwork{}{}.csv".format(plant, timePoint)
    knownPeripheralCellsFilename = folder + "periphery labels {}{}_control.csv".format(plant, timePoint)
    networkFilename = folder + filename
    folderToSave = "{}peripheralCellVerification/{}/".format(dataFolder, plant)
    Path(folderToSave).mkdir(parents=True, exist_ok=True)
    network = GraphCreatorFromAdjacencyList(networkFilename).GetGraph()
    # remove cells
    myPeripheralCellIdentifier = PeripheralCellIdentifier(network, tissueName=plant+timePoint, folderToSaveLog=folderToSave)
    peripheralCells = myPeripheralCellIdentifier.GetPeripheralCells()
    print("peripheralCells", peripheralCells)
    print(np.any(np.isin(peripheralCells, 125)))
    myPeripheralCellIdentifier.ValidateWithKnownCells(knownPeripheralCellsFilename, showPlot=showPlot)
    heatMapFilename = "{}peripheralHeatmap {}{} auto.csv".format(folderToSave, plant, timePoint)
    # myPeripheralCellIdentifier.ValidateCreatingHeatmap(knownPeripheralCellsFilename, heatMapFilename)

if __name__ == '__main__':
    main()
