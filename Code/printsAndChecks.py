import numpy as np
import pandas as pd
import pickle
import sys

def mainPrintXOutOfPossibleNonDivCellAfterPropagation():
    # check acutal out of potential non-dividing cells which were compared in propagation
    baseDataFolder = "Data/WT/divEventData/manualCentres/topology/"
    baseResultsFolder="Results/DivAndTopoApplication/"
    plantNames = ["P2", "P9"]
    fullLabelTable = pd.read_csv(baseDataFolder+"/combinedLabels.csv")
    fullPredictedFeatureOfNonDivCellsTable = []
    for plant in plantNames:
        predictedFeatureTable = np.load(baseResultsFolder+plant+"/predFeatures.npy")
        fullPredictedFeatureOfNonDivCellsTable.append(predictedFeatureTable)
    isPlant = np.isin(fullLabelTable["plant"], plantNames)
    fullLabelTable = fullLabelTable.iloc[isPlant, :]
    possibleNonDiv = np.sum(fullLabelTable["label"] == 0)
    fullPredictedFeatureOfNonDivCellsTable = np.concatenate(fullPredictedFeatureOfNonDivCellsTable, axis=0)
    actualNonDiv = len(fullPredictedFeatureOfNonDivCellsTable)
    print(f"{actualNonDiv} of the possible {possibleNonDiv} non-dividing cells in the observed topology were also non-dividing in the predicted topology")

if __name__== "__main__":
    mainPrintXOutOfPossibleNonDivCellAfterPropagation()
