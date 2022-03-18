import sys

sys.path.insert(0, "./Code/Analysis Tools/")
from BarPlotPlotter import mainDivPredRandomization
from DetailedCorrelationDataPlotter import DetailedCorrelationDataPlotter
from FeatureDensityPlotting import FeatureDensityPlotting
from GeneralDataAnalyser import GeneralDataAnalyser
from pathlib import Path
from VisualisePredictionsOnTissue import mainCreateTissuePredictionColoringOf

def plotAndPrepareMainFigures(resultsFolder="Results/MainFigures/", figuresToDo="Fig. 2 A"):
    Path(resultsFolder).mkdir(parents=True, exist_ok=True)
    # Fig. 2 A - .csv-file to use in MGX showing TP&TN in blue and FN&FP
    if figuresToDo == "all" or "Fig. 2 A" in figuresToDo:
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=True,
                    featureSetName="allTopos", saveUnderFolder=resultsFolder)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=True,
                    featureSetName="area", saveUnderFolder=resultsFolder)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=True,
                    featureSetName="topoAndBio", saveUnderFolder=resultsFolder)
    # Fig. 2 B - div prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 2 B" in figuresToDo:
        mainDivPredRandomization(performance="Acc", doMainFig=True,
                                 baseResultsFolder="Results/divEventData/manualCentres/",
                                 addOtherTestWithBaseFolder="Results/ktnDivEventData/manualCentres/",
                                 savePlotFolder=resultsFolder)
    # Fig. 3 A - .csv-file of correct and wrong topo predictions on example cell
    # Fig. 3 B - topo prediction acc. results to further add text in power point
    # Fig. 3 C - topo prediction AUC results of different classes for WT test
    # Fig. 3 C - topo prediction AUC results of different classes for ktn test
    # Fig. 4 B - example of predicted and observed topo prediction
    # Fig. 4 C - percentage of of correctly estimated neighbours per local topology
    # Fig. 4 D - concordance between the observed and predicted topologies
    # Fig. 4 E - comparison of obs. vs  pred. harmonic centrality applying div. and topo. prediction models

def plotAndPrepareSuppFigures(resultsFolder="Results/SuppFigures/"):
    Path(resultsFolder).mkdir(parents=True, exist_ok=True)
    # Fig. 2 - Pearson corr coeff of topological features vs area
    # Fig. 5 A - div prediction acc. randomisation results to further add text in power point
    # Fig. 5 B - topo prediction acc. randomisation results to further add text in power point
    # Fig. 6 A-C - topo prediction ROC curves results of different classes for WT test
    # Fig. 6 E-G - topo prediction ROC curves results of different classes for ktn test
    # Fig. 8 - comparison of obs. vs  pred. of all centrality applying div. and topo. prediction models
    # Fig. 9 - density distributions of features

if __name__== "__main__":
    plotAndPrepareMainFigures()
