import os
import networkx as nx
import numpy as np
import pandas as pd
import sys
modulePath = "./Code/"
sys.path.insert(0, modulePath)

from CellInSAMCenterDecider import CellInSAMCenterDecider
from utils import convertTextToLabels
from FeatureVectorCreator import FeatureVectorCreator
from FilenameCreator import FilenameCreator
from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from LabelCreator import LabelCreator
from scipy import stats
from TableCombiner import TableCombiner

class CreateFeaturesAndLabels (object):

    def __init__(self, baseFolder, plantNames, timePointsPerPlant, centerCellLabels=None,
                useEdgeWeight=False, usePlantNamesAsFolder=True,
                centerRadius=30, zNormaliseFeaturesPerTissue=False,
                secondNeighbourhoodProperties=True, skipSample=[],
                lengthOfTimeStep=1, useOwnFeatures=False, onlyDoArea=False, withArea=False,
                normaliseArea=False, useDividingCells=True, invertEdgeWeight=False,
                maxNormEdgeWeightPerGraph=False, useSharedWallWeight=False,
                removeSecondPeripheralCells=False):
        # creating features and labels time point successively
        self.baseFolder = baseFolder
        self.plantNames = plantNames
        self.timePointsPerPlant = timePointsPerPlant
        self.centerCellLabels = centerCellLabels
        self.useEdgeWeight = useEdgeWeight
        self.usePlantNamesAsFolder = usePlantNamesAsFolder
        self.centerRadius = centerRadius
        self.zNormaliseFeaturesPerTissue = zNormaliseFeaturesPerTissue
        self.secondNeighbourhoodProperties = secondNeighbourhoodProperties
        self.skipSample = skipSample
        self.tissueRanges = []
        self.timePointOfTissue = []
        self.plantNameOfTissue = []
        self.lengthOfTimeStep = lengthOfTimeStep
        self.useOwnFeatures = useOwnFeatures
        self.onlyDoArea = onlyDoArea
        self.withArea = withArea
        self.normaliseArea = normaliseArea
        self.useDividingCells = useDividingCells
        self.invertEdgeWeight = invertEdgeWeight
        self.maxNormEdgeWeightPerGraph = maxNormEdgeWeightPerGraph
        self.useSharedWallWeight = useSharedWallWeight
        self.removeSecondPeripheralCells = removeSecondPeripheralCells
        self.allPossibleLabels = []
        self.propertyOfTissues = None
        self.setFilenames(self.baseFolder, self.timePointsPerPlant, self.plantNames,
                        lengthOfTimeStep=self.lengthOfTimeStep,
                        usePlantNamesAsFolder=self.usePlantNamesAsFolder)
        self.allFeatures, self.allLabels = self.calcAllFeaturesAndLabels(self.timePointsPerPlant)
        self.combinedFeatureVector = TableCombiner(self.allFeatures, isParameterFilename=False).GetMergedData()
        if self.useEdgeWeight is False and onlyDoArea is False:
            self.combinedFeatureVector.drop("weighted node degree", axis=1, inplace=True)
        self.combinedLabelVector = TableCombiner(self.allLabels, isParameterFilename=False).GetMergedData()

    def setFilenames(self, baseFolder, timePointsPerPlant, plantNames, lengthOfTimeStep=1,
                     usePlantNamesAsFolder=True):
        myFilenameCreator = FilenameCreator(baseFolder, timePointsPerPlant, plantNames,
                    usePlantNamesAsFolder=usePlantNamesAsFolder, lengthOfTimeStep=lengthOfTimeStep,
                    connectivityText="cellularConnectivityNetwork",
                    parentLabelingText="parentLabeling", geometryText="area",
                    peripheralLabelText="periphery labels ")
        self.connectivityNetworkFilenames = myFilenameCreator.GetConnectivityNetworkFilenames()
        self.areaFilenames = myFilenameCreator.GetAreaFilenames()
        self.peripheralLabelsFilename = myFilenameCreator.GetPeripheralLabelsFilenames()
        self.parentLabelingFilenames = myFilenameCreator.GetPartentLabellingFilenames()

    def calcAllFeaturesAndLabels(self, timePointsPerPlant):
        allFeatures = []
        allLabels = []
        plantNameOfCell = []
        timeOfCell = []
        labelOfCell = []
        fileIdx = 0
        for timeIdx in range(timePointsPerPlant-1):
            for plantIdx in range(len(self.plantNames)):
                if not fileIdx in self.skipSample:
                    featureVector, labelVector = self.calcFeatureVectorsAndLabelsOf(fileIdx, plantIdx, timeIdx)
                    allFeatures.append(featureVector)
                    allLabels.append(labelVector)
                    nrOfSamples = len(featureVector)
                    plantNameOfCell.append(np.full(nrOfSamples, plantIdx))
                    timeOfCell.append(np.full(nrOfSamples, timeIdx))
                    labelOfCell.append(list(featureVector.index))
                    self.timePointOfTissue.append(timeIdx)
                    self.plantNameOfTissue.append(self.plantNames[plantIdx])
                fileIdx += 1
        propertyOfTissues = {"plant":np.concatenate(plantNameOfCell),
                             "timePoint":np.concatenate(timeOfCell),
                             "label":np.concatenate(labelOfCell)}
        self.propertyOfTissues = pd.DataFrame(propertyOfTissues)
        return allFeatures, allLabels

    def calcFeatureVectorsAndLabelsOf(self, fileIdx, plantIdx, timeIdx):
        cellSizeFilename = self.areaFilenames[fileIdx]
        currentFilename = self.connectivityNetworkFilenames[fileIdx]
        graphCreator = GraphCreatorFromAdjacencyList(currentFilename,
                                        cellSizeFilename=cellSizeFilename,
                                        useEdgeWeight=self.useEdgeWeight,
                                        invertEdgeWeight=self.invertEdgeWeight,
                                        useSharedWallWeight=self.useSharedWallWeight)
        self.currentGraph = graphCreator.GetGraph()
        if self.maxNormEdgeWeightPerGraph and self.useEdgeWeight:
            max = 0
            for u,v,d in self.currentGraph.edges(data=True):
                if d['weight'] > max:
                    max = d['weight']
            for u,v,d in self.currentGraph.edges(data=True):
                d['weight'] /= max
        allowedLabels = self.defineValidLabels(fileIdx, plantIdx, timeIdx)
        if self.onlyDoArea:
            table = pd.read_csv(cellSizeFilename, index_col="Label", skipfooter=4, engine="python")
            selectedLabels = np.isin(list(table.index), allowedLabels)
            featureVector = graphCreator.GetCellSizes().to_frame()
            featureVector = featureVector.iloc[selectedLabels, :].copy()
            if self.normaliseArea:
                featureVector.iloc[:] = stats.zscore(featureVector.iloc[:])
            featureVector.columns = ["area"]
        else:
            featureVector = FeatureVectorCreator(self.currentGraph, allowedLabels=allowedLabels,
                            zNormaliseFeaturesPerTissue=self.zNormaliseFeaturesPerTissue,
                            secondNeighbourhoodProperties=self.secondNeighbourhoodProperties,
                            useOwnFeatures=self.useOwnFeatures,
                            filename=currentFilename).GetFeatureMatrixDataFrame()
            if self.withArea:
                areas = graphCreator.GetCellSizes()
                selectedLabels = np.isin(list(areas.index), list(featureVector.index))
                areas = areas.iloc[selectedLabels].copy()
                if self.normaliseArea:
                    areas = stats.zscore(areas)
                featureVector.insert(0, 'area', areas)
        if self.useDividingCells:
            sampleName = "{}T{}T{}".format(self.plantNames[plantIdx], timeIdx, timeIdx+1)
            additionalDividingCellsFilename = "{}{}/dividingCells{}.txt".format(self.baseFolder, self.plantNames[plantIdx], sampleName)
            if not os.path.isfile(additionalDividingCellsFilename):
                additionalDividingCellsFilename = None
        else:
            additionalDividingCellsFilename = None
        labelVector = LabelCreator(self.parentLabelingFilenames[fileIdx], featureVector,
                                allowedLabels=allowedLabels, areParametersFilenames=[True,False],
                                useDividingCells=additionalDividingCellsFilename).GetLabelVector()
        self.testFeaturesLabelsAndCentralRegions(allowedLabels, featureVector, labelVector)
        self.tissueRanges.append(len(labelVector))
        return featureVector, labelVector

    def defineValidLabels(self, fileIdx, plantIdx, timeIdx):
        centralRegionLabels = self.setCentralRegionLabel(fileIdx, plantIdx, timeIdx)
        peripheralLabels = convertTextToLabels(self.peripheralLabelsFilename[fileIdx],
                                allLabelsFilename=self.areaFilenames[fileIdx]).GetLabels(onlyUnique=True)
        peripheralLabels = peripheralLabels.astype(int)
        if self.secondNeighbourhoodProperties or self.removeSecondPeripheralCells:
            peripheralLabels = self.addNeighboringLabelsOfCurrentGraph(peripheralLabels)
        allowedLabels = self.furtherFilterCentralRegions(centralRegionLabels, peripheralLabels)
        additionalLabelsToRemoveFilename = self.baseFolder + "{}/{}{}T{}.txt".format(self.plantNames[plantIdx], "additionalLabelsToRemove", self.plantNames[plantIdx], timeIdx)
        if os.path.isfile(additionalLabelsToRemoveFilename):
            additionalLabelsToRemove = convertTextToLabels(additionalLabelsToRemoveFilename,
                                                allLabelsFilename=self.areaFilenames[fileIdx]).GetLabels(onlyUnique=True)
            allowedLabels = self.furtherFilterCentralRegions(allowedLabels, additionalLabelsToRemove)
        return allowedLabels

    def setCentralRegionLabel(self, fileIdx, plantIdx, timeIdx):
        if not self.centerCellLabels is None:
            currentCenterCellLabels = self.centerCellLabels[plantIdx][timeIdx]
            myCellInSAMCenterDecider = CellInSAMCenterDecider(self.areaFilenames[fileIdx],
                                        currentCenterCellLabels, centerRadius=self.centerRadius)
            centralRegionLabels = myCellInSAMCenterDecider.GetCentralCells()
        else:
            centralRegionLabels = None
        return centralRegionLabels

    def addNeighboringLabelsOfCurrentGraph(self, peripheralLabels):
        adjacencyList = nx.to_dict_of_dicts(self.currentGraph)
        additionalLabels = []
        for label in peripheralLabels:
            if label in adjacencyList:
                additionalLabels.extend(list(adjacencyList[label].keys()))
        additionalLabels = np.unique(additionalLabels)
        addionalPeripheryCells = np.concatenate([peripheralLabels, additionalLabels])
        return np.unique(addionalPeripheryCells)

    def furtherFilterCentralRegions(self, allowedLabels, labelsToRemove):
        isLabelRemaining = np.isin(allowedLabels, labelsToRemove, invert=True)
        allowedLabels = allowedLabels[isLabelRemaining]
        return allowedLabels

    def allSummedSecondNeighbourhoods(self, graph, selectedNodes=None):
        summedSecondNeighbourhoods = []
        adjacencyList = nx.to_dict_of_dicts(graph)
        if selectedNodes is None:
            selectedNodes = graph.nodes()
        for node in selectedNodes:
            summedSecondNeighbourhoods.append(self.caclulateSummedSecondNeighborhood(node, adjacencyList))
        return summedSecondNeighbourhoods

    def caclulateSummedSecondNeighborhood(self, node, adjacencyList):
        neighbouringNodes = adjacencyList[node].keys()
        summedNeighbours = 0
        for neighbour in neighbouringNodes:
            summedNeighbours += len(list(adjacencyList[neighbour].keys()))
        return summedNeighbours

    def testFeaturesLabelsAndCentralRegions(self, centralRegionLabels, featureVector, labelVector):
        if not self.centerCellLabels is None:
            if len(centralRegionLabels) == 0:
                print("ATTENTION: The number of central region labels is zero.")
            #assert len(centralRegionLabels) == len(featureVector), print("Number of central region labels is not equal to the number of feature labels. {} != {}".format(len(centralRegionLabels), len(featureVector)))
            #assert len(centralRegionLabels) == len(labelVector), print("Number of central region labels is not equal to the number of class labels. {} != {}".format(len(centralRegionLabels), len(labelVector)))
        assert len(labelVector) == len(featureVector), print("Number of feature labels is not equal to the number of class labels. {} != {}".format(len(labelVector), len(featureVector)))

    def GetCombinedFeatureMatrixAsDataFrame(self):
        return self.combinedFeatureVector

    def GetCombinedLabels(self):
        return self.combinedLabelVector

    def GetPropertyOfTissues(self):
        return self.propertyOfTissues

    def GetNrOfSamplesPerTissue(self):
        return self.tissueRanges

    def GetTimePointOfTissue(self):
        return np.asarray(self.timePointOfTissue)

    def GetReplicateId(self):
        return np.asarray(self.plantNameOfTissue)

    def GetAreaFilenames(self):
        return self.areaFilenames

    def SaveFeaturesAndLabels(self, featureFilename="", labelFilename="", resultsFolder="", sep=","):
        if resultsFolder:
            featureFilename = resultsFolder+"combinedFeatures.csv"
            labelFilename = resultsFolder+"combinedLabels.csv"
        if featureFilename and labelFilename:
            self.combinedFeatureVector.to_csv(featureFilename ,sep=sep)
            self.combinedLabelVector.to_csv(labelFilename ,sep=sep, header=["dividing cells"])
        else:
            print("Nothing was saved, because either the filename of the feature or labels vector was not correct")

def main():
    baseFolder = "./Data/WT/"
    useEdgeWeight = True
    zNormaliseFeaturesPerTissue = True
    secondNeighbourhoodProperties = True
    onlyDoArea = True
    withArea = True
    centerRadius = 30
    useOwnFeatures = ["degree"]
    skipSample = []#[8, 13, 14, 19]
    plantNames = ["P5"]#["P1","P2", "P5", "P6", "P8"]
    centerCellLabels = [#[[618, 467, 570], [5048, 5305], [5380, 5849, 5601], [6178, 6155, 6164]],
                        #[[392],  [553, 779, 527], [525], [1135]],
                        [[38], [585, 968, 982], [927, 1017], [1136]]]#,
                        # [[861], [1651, 1621], [1763, 1844], [2109, 2176]],
                        # [[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013]]]
    skipSample = [8, 13, 14, 19]
    plantNames = ["P1","P2", "P5", "P6", "P8"]
    centerCellLabels = [[[618, 467, 570], [5048, 5305], [5380, 5849, 5601], [6178, 6155, 6164]],
                        [[392],  [553, 779, 527], [525], [1135]],
                        [[38], [585, 968, 982], [927, 1017], [1136]],
                        [[861], [1651, 1621], [1763, 1844], [2109, 2176]],
                        [[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013]]]
    additionalNumber = ""
    centerCellLabelsText = ""
    byCellSizeWeightedEdgesText = ""
    zNormaliseFeaturesPerTissueText = "_notNorm"
    if not centerCellLabels is None:
        centerCellLabelsText = "r"
        if centerRadius:
            centerCellLabelsText += str(centerRadius)
        if useEdgeWeight is True:
            centerCellLabelsText += "_"
    if useEdgeWeight is True:
        byCellSizeWeightedEdgesText = "WeightedEdges"
    if zNormaliseFeaturesPerTissue is True:
        zNormaliseFeaturesPerTissueText = "_zNormTissueWise"
    dataSetProperties = "{}{}{}{}".format(centerCellLabelsText, byCellSizeWeightedEdgesText, zNormaliseFeaturesPerTissueText, additionalNumber)
    featureFilename = baseFolder + "features_plusP1_{}.csv".format(dataSetProperties)
    labelFilename = baseFolder + "labels_plusP1_{}.csv".format(dataSetProperties)
    print(labelFilename)
    print(featureFilename)
    # centerCellLabels need to be given as plant/repilcate per row and time point per column
    myCreateFeaturesAndLabels = CreateFeaturesAndLabels(baseFolder, plantNames,
                                            5, centerCellLabels=centerCellLabels,
                                            useEdgeWeight=useEdgeWeight,
                                            centerRadius=centerRadius,
                                            zNormaliseFeaturesPerTissue=zNormaliseFeaturesPerTissue,
                                            secondNeighbourhoodProperties=secondNeighbourhoodProperties,
                                            skipSample=skipSample,
                                            onlyDoArea=onlyDoArea,
                                            withArea=withArea,
                                            useOwnFeatures=useOwnFeatures,
                                            removeSecondPeripheralCells=True)
    combinedLabels = myCreateFeaturesAndLabels.GetCombinedLabels()
    propertyOfTissues = myCreateFeaturesAndLabels.GetPropertyOfTissues()
    print(propertyOfTissues)
    sys.exit()
    print(np.unique(combinedLabels, return_counts=True))
    # print(myCreateFeaturesAndLabels.GetNrOfSamplesPerTissue())
    print(myCreateFeaturesAndLabels.GetCombinedFeatureMatrixAsDataFrame().head())
    print(myCreateFeaturesAndLabels.GetCombinedLabels().head())


    #myCreateFeaturesAndLabels.SaveFeaturesAndLabels(featureFilename=featureFilename, labelFilename=labelFilename) #resultsFolder=baseFolder)

if __name__ == '__main__':
    main()
