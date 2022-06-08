import glob
import os
import numpy as np
import pandas as pd

class GetDividingAndNonDividingLabels (object):

    def __init__(self, topologyFilenames, parentFilename, givePattern=True):
        self.sep = ","
        self.givePattern = givePattern
        self.setupTables(topologyFilenames, parentFilename)

    def setupTables(self, topologyFilenames, parentFilename):
        if self.givePattern:
            self.topologyFilenames = glob.glob(topologyFilenames)
        else:
            self.topologyFilenames = topologyFilenames
        self.topologyTables = self.loadTables(self.topologyFilenames)
        if self.givePattern:
            self.parentFilename = glob.glob(parentFilename)
        else:
            self.parentFilename = parentFilename
        self.parentTables = self.loadTables(self.parentFilename)

    def loadTables(self, filenames):
        tables = []
        for filename in filenames:
            table = pd.read_csv(filename, sep=self.sep)
            tables.append(table)
        return tables

    def ExtractDividingCells(self, selectedLayer=None):
        self.dividingCellsPerLayer = {}
        self.nrOfCellsPerLayer = {}
        for i, parentalDf in enumerate(self.parentTables):
            currentParentalDfFilename = self.topologyFilenames[i]
            layerDf = self.topologyTables[i]
            currentTissuesDividingCellsAndLayer = self.extractDividingCells(parentalDf, layerDf)
            self.dividingCellsPerLayer = self.combineDicts(self.dividingCellsPerLayer, currentTissuesDividingCellsAndLayer)
            if "T0" in currentParentalDfFilename:
                nNumbersOfCellsPerTissue = self.extractNrOfCellsPerTissueLayer(layerDf)
                self.nrOfCellsPerLayer = self.combineDicts(self.nrOfCellsPerLayer, nNumbersOfCellsPerTissue)
        if selectedLayer:
            self.printDivdidingCellsPerLayer(selectedLayer, self.dividingCellsPerLayer, self.nrOfCellsPerLayer)
        else:
            for selectedLayer in self.dividingCellsPerLayer.keys():
                self.printDivdidingCellsPerLayer(selectedLayer, self.dividingCellsPerLayer, self.nrOfCellsPerLayer)

    def extractDividingCells(self, parentalDf, layerDf):
        parentalCells = parentalDf.loc[:, " Parent Label"]
        uniqueParents, counts = np.unique(parentalCells, return_counts=True)
        dividingCells = uniqueParents[counts > 1]
        isDivCellInLayerDf = np.isin(layerDf.loc[:, "Label"], dividingCells)
        layerOfDividingCells = layerDf.loc[isDivCellInLayerDf, "Layer"]
        currentTissuesDividingCellsAndLayer = {}
        for layer in np.unique(layerOfDividingCells):
            dividingCellsOfCurrentLayer = dividingCells[layerOfDividingCells==layer]
            currentTissuesDividingCellsAndLayer[layer] = list(dividingCellsOfCurrentLayer)
        return currentTissuesDividingCellsAndLayer

    def combineDicts(self, dict1, dict2):
        for key, values in dict2.items():
            assert isinstance(values, list), "The values {} of key {} in dict2 {} need to be list, {} != list".format(dict2, key, type(values))
            if key in dict1:
                assert isinstance(dict1[key], list), "The values {} of key {} in dict1 {} need to be list, {} != list".format(dict1, key, type(dict1[key]))
                dict1[key].extend(values)
            else:
                dict1[key] = values
        return dict1

    def extractNrOfCellsPerTissueLayer(self, layerDf):
        nrOfCellsPerTissueLayer = {}
        for layer in np.unique(layerDf.loc[:, "Layer"]):
            nrOfCellsPerTissueLayer[layer] = list(layerDf.loc[layerDf.loc[:, "Layer"]==layer, "Label"])
        return nrOfCellsPerTissueLayer

    def printDivdidingCellsPerLayer(self, selectedLayer, dividingCellsPerLayer, nrOfCellsPerLayer):
        nrOfDividingCells = len(dividingCellsPerLayer[selectedLayer])
        totalCellNumber =  len(nrOfCellsPerLayer[selectedLayer])
        percentage = np.round(100*nrOfDividingCells/totalCellNumber, 1)
        print("In layer: {}, {} / {} divided ({}%)".format(selectedLayer, nrOfDividingCells, totalCellNumber, percentage))

    def printDividingCellsOverAllLayers(self, dividingCellsPerLayer, nrOfCellsPerLayer):
        nrOfDividingCells = len(np.concatenate([dividingCellsPerLayer[selectedLayer] for selectedLayer in dividingCellsPerLayer.keys()]))
        totalCellNumber =  len(np.concatenate([nrOfCellsPerLayer[selectedLayer] for selectedLayer in nrOfCellsPerLayer.keys()]))
        percentage = np.round(100*nrOfDividingCells/totalCellNumber, 1)
        print("In layer: {}, {} / {} divided ({}%)".format(list(dividingCellsPerLayer.keys()), nrOfDividingCells, totalCellNumber, percentage))

    def GetDividingCellsPerLayer(self):
        return self.dividingCellsPerLayer

    def GetNrOfCellsPerLayer(self):
        return self.nrOfCellsPerLayer

def main():
    # print number of dividing cells of all cells in Jackson et al., 2019
    # I found out that the label dividing is propagated
    # meaning cells dividing in the first time point, e.g. label 362 (T0), splits into 1276 and 292 (T1),
    # but in the file "02_T1_topology.csv" the cells 1276 and 292 are marked with 1 in the Division column
    # therefore I check the parents labels of e.g. "02_T0-T1_parents.csv", "02_T1-T2_parents.csv" for multiple entries
    # download 'WT Arabidopsis.zip' data from https://osf.io/h587k/ (see STAR★Methods of Jackson et al., 2019)
    # and copy this script into the base folder (WT Arabidopsis) containing the 'labels.csv' and 'README.txt' file
    # and execute the script
    import sys
    baseFolderOfData = "Jackson Data WT Arabidopsis"
    filenameToSave = "labels.csv"
    nrOfCellsPerLayer, dividingCellsPerLayer = {}, {}
    for i in range(1,5):
        print("Replicate {}".format(i))
        replicateFolder = "{}/Rep {}/*/".format(baseFolderOfData, i)
        topologyPatterns = replicateFolder + "*topology.csv"
        parentPatterns = replicateFolder + "*parents.csv"
        myGetDividingAndNonDividingLabels = GetDividingAndNonDividingLabels(topologyPatterns, parentPatterns)
        myGetDividingAndNonDividingLabels.ExtractDividingCells()
        currentDividingCellsPerLayer = myGetDividingAndNonDividingLabels.GetDividingCellsPerLayer()
        currentNrOfCellsPerLayer = myGetDividingAndNonDividingLabels.GetNrOfCellsPerLayer()
        dividingCellsPerLayer = myGetDividingAndNonDividingLabels.combineDicts(dividingCellsPerLayer, currentDividingCellsPerLayer)
        nrOfCellsPerLayer = myGetDividingAndNonDividingLabels.combineDicts(nrOfCellsPerLayer, currentNrOfCellsPerLayer)
    print("all replicates summariesed:")
    for selectedLayer in dividingCellsPerLayer.keys():
        myGetDividingAndNonDividingLabels.printDivdidingCellsPerLayer(selectedLayer, dividingCellsPerLayer, nrOfCellsPerLayer)
    myGetDividingAndNonDividingLabels.printDividingCellsOverAllLayers(dividingCellsPerLayer, nrOfCellsPerLayer)


if __name__ == '__main__':
    main()
