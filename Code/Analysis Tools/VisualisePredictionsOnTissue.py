import numpy as np
import pandas as pd
import pickle
import sys
from sklearn import metrics

modulePath = "./Code/Classifiers/"
sys.path.insert(0, modulePath)
from NestedModelCreator import NestedModelCreator
from pathlib import Path


class VisualisePredictionsOnTissue (object):

    # class to create heatmap to visualise correct and wrong predictions (or confusion matrix) on a tissue
    def __init__(self, geometryTableFilename, cellObsAndPredLabelsDf,
                       filenameToSave=None, toSavesuffix="_predictionVisualisation.csv",
                       valueToColor={"TP": 1, "TN": 2, "FP": 3, "FN": 0},
                       colNames=["cellId", "observedLabels", "predictedLabels"]):
        self.geometryTableFilename = geometryTableFilename
        self.geometryDf = pd.read_csv(self.geometryTableFilename, skipfooter=4, index_col=0, engine="python")
        self.cellId = cellObsAndPredLabelsDf.loc[:, colNames[0]]
        self.trueLabels = cellObsAndPredLabelsDf.loc[:, colNames[1]]
        self.predictedLabels = cellObsAndPredLabelsDf.loc[:, colNames[2]]
        self.filenameToSave = filenameToSave
        self.toSavesuffix = toSavesuffix
        self.positiveLabel = 1
        self.neagtiveLabel = 0
        self.valueToColor = valueToColor

    def savePredictionAsHeatMap(self):
        self.removeAndOrderGeometryDf()
        self.determineTPandTNandFPandFN()
        self.colorEntries()
        self.saveTable()

    def removeAndOrderGeometryDf(self):
        allCellLabels = self.geometryDf.index.to_numpy()
        newOrder = np.zeros(len(self.cellId))
        notPresentCells = {}
        for i, cellId in enumerate(self.cellId):
            isCurrentId = np.isin(allCellLabels, cellId)
            if np.sum(isCurrentId) == 1:
                newOrder[i] = np.where(isCurrentId)[0][0]
            else:
                notPresentCells[cellId] = np.sum(isCurrentId)
        self.reducedGeometryDf = self.geometryDf.iloc[newOrder, :].copy()
        assert len(notPresentCells) == 0, "cells present X times, {}".format(notPresentCells)

    def determineTPandTNandFPandFN(self):
        correctPredicted = self.trueLabels == self.predictedLabels
        dividingCells = self.trueLabels == self.positiveLabel
        self.TP = np.where(correctPredicted & dividingCells)[0]
        self.TN = np.where(correctPredicted & np.invert(dividingCells))[0]
        self.FP = np.where(np.invert(correctPredicted) & np.invert(dividingCells))[0]
        self.FN = np.where(np.invert(correctPredicted) & dividingCells)[0]

    def colorEntries(self):
        self.reducedGeometryDf.iloc[self.TP, 0] = self.valueToColor["TP"]
        self.reducedGeometryDf.iloc[self.TN, 0] = self.valueToColor["TN"]
        self.reducedGeometryDf.iloc[self.FP, 0] = self.valueToColor["FP"]
        self.reducedGeometryDf.iloc[self.FN, 0] = self.valueToColor["FN"]

    def saveTable(self):
        if self.filenameToSave is None:
            self.filenameToSave = self.geometryTableFilename[:-4] + self.toSavesuffix
        self.reducedGeometryDf.to_csv(self.filenameToSave, index_label="Label")
        lastLines = self.extractLastLines(self.geometryTableFilename, extractFooter=4)
        self.appendLinesToFile(lastLines, self.filenameToSave)
        print("saved", self.filenameToSave)

    def extractLastLines(self, fileToOpen, extractFooter=0):
        lastLines = ""
        file = open(fileToOpen, "r")
        allLines = file.readlines()
        file.close()
        if int(extractFooter) > 0:
            if len(allLines) >= extractFooter:
                lastLines = allLines[-extractFooter:]
                lastLines = "".join(lastLines)
                # do I need to change the range or the mesh number?
            else:
                print("The number of footer lines to extract ({}) was larger than the number of lines in the file {}.".format(extractFooter, fileToOpen))
        return lastLines

    def appendLinesToFile(self, lastLines, saveToFilename):
        file = open(saveToFilename, "a")
        file.write(lastLines)
        file.close()

class VisualiseTopologyPredictionsOnTissue (VisualisePredictionsOnTissue):

    def __init__(self, geometryTableFilename, cellObsAndPredLabelsDf,
                       filenameToSave=None, toSavesuffix="_predictionVisualisation.csv",
                       valueToColor={"TP": 1, "TN": 2, "FP": 3, "FN": 0},
                       colNames=["cellId", "observedLabels", "predictedLabels", "dividingCellId"]):
        super().__init__(geometryTableFilename=geometryTableFilename, cellObsAndPredLabelsDf=cellObsAndPredLabelsDf,
                           filenameToSave=filenameToSave, toSavesuffix=toSavesuffix,
                           valueToColor=valueToColor,
                           colNames=colNames)
        self.dividingCellId = cellObsAndPredLabelsDf.loc[:, colNames[3]]

    def savePredictionAsHeatMap(self, folder=None):
        for dividingCell in np.unique(self.dividingCellId):
            self.removeAndOrderGeometryDf(dividingCell)
            self.determineTPandTNandFPandFN(dividingCell)
            self.colorEntries()
            self.saveTable(dividingCell, folder)

    def removeAndOrderGeometryDf(self, dividingCell):
        allCellLabels = self.geometryDf.index.to_numpy()
        neighbourCells = self.cellId[self.dividingCellId == dividingCell]
        newOrder = np.zeros(len(neighbourCells))
        notPresentCells = {}
        for i, cellId in enumerate(neighbourCells):
            isCurrentId = np.isin(allCellLabels, cellId)
            if np.sum(isCurrentId) == 1:
                newOrder[i] = np.where(isCurrentId)[0][0]
            else:
                notPresentCells[cellId] = np.sum(isCurrentId)
        self.reducedGeometryDf = self.geometryDf.iloc[newOrder, :].copy()
        assert len(notPresentCells) == 0, "cells present X times, {}".format(notPresentCells)

    def determineTPandTNandFPandFN(self, dividingCell):
        selectedCells = self.reducedGeometryDf.index.to_numpy()
        isSelectedCell = np.isin(self.cellId, selectedCells)
        isDividingCell = self.dividingCellId == dividingCell
        isSelectedCell = isSelectedCell & isDividingCell
        correctPredicted = self.trueLabels[isSelectedCell] == self.predictedLabels[isSelectedCell]
        dividingCells = self.trueLabels[isSelectedCell] == self.positiveLabel
        self.TP = np.where(correctPredicted & dividingCells)[0]
        self.TN = np.where(correctPredicted & np.invert(dividingCells))[0]
        self.FP = np.where(np.invert(correctPredicted) & np.invert(dividingCells))[0]
        self.FN = np.where(np.invert(correctPredicted) & dividingCells)[0]

    def saveTable(self, dividingCell, folder=None):
        if folder is None:
            self.filenameToSave = "{}_around{}_{}".format(self.geometryTableFilename[:-4], dividingCell, self.toSavesuffix)
        else:
            self.filenameToSave = "{}{}_around{}_{}".format(folder, Path(self.geometryTableFilename[:-4]).stem, dividingCell, self.toSavesuffix)
        self.reducedGeometryDf.to_csv(self.filenameToSave, index_label="Label")
        lastLines = self.extractLastLines(self.geometryTableFilename, extractFooter=4)
        self.appendLinesToFile(lastLines, self.filenameToSave)
        print("saved", len(self.reducedGeometryDf), self.filenameToSave)

class FeatureLabelAndNameSelecor (object):

    def __init__(self, featureArray, labelArray, combinedLabelsDf, model):
        self.featureArray = featureArray
        self.labelArray = labelArray
        self.originalCombinedLabelsDf = combinedLabelsDf.copy()
        self.model = model

    def extractAndPredictLabels(self, plantName, timePoint, onlyExtractCell=True, reduceToTest=None):
        if reduceToTest is None:
            self.combinedLabelsDf = self.originalCombinedLabelsDf.copy()
        else:
            isCellSelected = self.originalCombinedLabelsDf.loc[:, "isCellTest"]
            if not reduceToTest:
                isCellSelected = np.invert(isCellSelected)
            self.combinedLabelsDf = self.originalCombinedLabelsDf[isCellSelected].copy()
        allPlantNames = self.combinedLabelsDf.loc[:, "plant"].to_numpy()
        isPlantName = allPlantNames == plantName
        assert np.sum(isPlantName) > 0, f"The plant name of the tissue plant name '{plantName}' / time point '{timePoint}' does not exist in the combinedLabelsDf, {plantName} not in {np.unique(allPlantNames)}"
        plantsDf = self.combinedLabelsDf.loc[isPlantName, :]
        allTimePoints = plantsDf.loc[:, "time point"].to_numpy()
        isTimePoint = allTimePoints == timePoint
        assert np.sum(isTimePoint) > 0, f"The time point of the tissue plant name '{plantName}' / time point '{timePoint}' does not exist in the combinedLabelsDf, {timePoint} not in {np.unique(allTimePoints)}"
        allTimePoints = self.combinedLabelsDf.loc[:, "time point"].to_numpy()
        isTimePoint = allTimePoints == timePoint
        isTissue = isPlantName & isTimePoint
        selectedFeatures = self.featureArray[isTissue, :]
        observedLabels = self.labelArray[isTissue]
        fullDfObservedLabels = self.combinedLabelsDf.loc[isTissue, "label"].to_numpy()
        assert len(fullDfObservedLabels) == len(observedLabels), "Observed label lengths is different."
        assert np.all(fullDfObservedLabels == observedLabels), "Observed labels are not all the same"
        predictedLabels = self.model.predict(selectedFeatures)
        if onlyExtractCell:
            cellId = self.combinedLabelsDf.loc[isTissue, "cell"].to_numpy()
            cellObsAndPredLabels = {"cellId":cellId,
                                    "observedLabels":observedLabels,
                                    "predictedLabels":predictedLabels}
        else:
            cellId = plantsDf.loc[isTissue, "parent neighbor"].to_numpy()
            dividingCellId = plantsDf.loc[isTissue, "dividing parent cell"].to_numpy()
            cellObsAndPredLabels = {"cellId":cellId,
                                    "dividingCellId":dividingCellId,
                                    "observedLabels":observedLabels,
                                    "predictedLabels":predictedLabels}
        self.cellObsAndPredLabelsDf = pd.DataFrame(cellObsAndPredLabels)

    def GetCellsObsAndPredDf(self):
        return self.cellObsAndPredLabelsDf

def mainCreateTissuePredictionColoringOf(doDivPredVisualisation=False,
            plantName="P2", timePoint=0, featureSetName=None, featureSetIdx=0,
            saveUnderFolder="", baseDataFolder="Data/WT/", colorSchemeIdx=-1):
    colorScheme = ["Confusion_Matrix_1", "Confusion_Matrix_2",
                   "Confusion_Matrix_3", "True_False_1"][colorSchemeIdx]
    if colorScheme == "Confusion_Matrix_1":
        valueToColor = {"TP": 1, "TN": 2, "FP": 3, "FN": 0}
    elif colorScheme == "Confusion_Matrix_2":
            valueToColor = {"TP": 0.5, "TN": 0.3, "FP": 0.95, "FN": 0.1}
    elif colorScheme == "Confusion_Matrix_3":
            valueToColor = {"TP": 0.1, "TN": 0.95, "FP": 0.7, "FN": 0.3}
    elif colorScheme == "True_False_1":
        valueToColor = {"TP": 0.1, "TN": 0.1, "FP": 0.95, "FN": 0.95}
    else:
        print("The color scheme {} does not exist use an exisiting color scheme or implement your own.")
        sys.exit()
    if doDivPredVisualisation:
        baseResultsFolder = "Results/divEventData/manualCentres/{}/svm_k2h_combinedTable_l3f0n1c0bal0ex0/"
        allFeatureSets = ["allTopos", "area", "topoAndBio", "lowCor0.7", "lowCor0.3", "topology"]
    else:
        baseResultsFolder = "Results/topoPredData/diff/manualCentres/{}/svm_k2h_combinedTable_l3f0n1c0bal0ex1/"
        allFeatureSets = ["allTopos", "bio", "topoAndBio", "lowCor0.7", "lowCor0.3", "topology"]
    if featureSetName is None:
        featureSetName = allFeatureSets[featureSetIdx]
    else:
        featureSetName
    resultsFolder = baseResultsFolder.format(featureSetName)
    geometryTableFilename = "{}{}/area{}T{}.csv".format(baseDataFolder, plantName, plantName, timePoint)
    filenameToSave = saveUnderFolder + "visualisingPredictionsOn_{}T{}_{}_with_{}.csv".format(plantName, timePoint, featureSetName, colorScheme)
    # read files
    combinedLabelsDf = pd.read_csv(resultsFolder + "labelOverviewDf.csv")
    featureArray = pd.read_csv(resultsFolder + "normalizedFeatures_test.csv").to_numpy()
    labelArray = pd.read_csv(resultsFolder + "labels_test.csv").to_numpy().flatten()
    model = pickle.load(open(resultsFolder + "testModel.pkl", "rb"))
    # combine cell id, observed and predicted labels
    selector = FeatureLabelAndNameSelecor(featureArray, labelArray,
                                          combinedLabelsDf, model)
    selector.extractAndPredictLabels(plantName, timePoint, onlyExtractCell=doDivPredVisualisation,
                                     reduceToTest=True)
    cellObsAndPredLabelsDf = selector.GetCellsObsAndPredDf()
    # visualise prediction results in geometry table
    if doDivPredVisualisation:
        visualiser = VisualisePredictionsOnTissue(geometryTableFilename,
                                              cellObsAndPredLabelsDf,
                                              filenameToSave=filenameToSave,
                                              valueToColor=valueToColor)
        visualiser.savePredictionAsHeatMap()
    else:
        toSavesuffix = "_predictionVisualisation_{}.csv".format(featureSetName)
        visualiser = VisualiseTopologyPredictionsOnTissue(geometryTableFilename,
                                              cellObsAndPredLabelsDf,
                                              filenameToSave=filenameToSave,
                                              valueToColor=valueToColor,
                                              toSavesuffix=toSavesuffix)
        visualiser.savePredictionAsHeatMap(saveUnderFolder)

if __name__ == '__main__':
    mainCreateTissuePredictionColoringOf()
