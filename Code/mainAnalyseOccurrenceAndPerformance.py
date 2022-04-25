import sys

sys.path.insert(0, "./Code/Analysis Tools/")
from BarPlotPlotter import mainDivPredRandomization, mainTopoPredRandomization
from BarPlotPlotterExtended import mainDivPredTestComparisons, mainTopoPredTestComparisons
from CorrelationHeatMapDisplayer import mainSaveCorrelation
from DetailedCorrelationDataPlotter import DetailedCorrelationDataPlotter
from FeatureDensityPlotting import mainSaveDensityPlotsOfFeaturesFromDiffScenarios
from GeneralDataAnalyser import GeneralDataAnalyser
from MyScorer import mainPlotRocCurvesAndAUCLabelDetails
from pathlib import Path
from VisualisePredictionsOnTissue import mainCreateTissuePredictionColoringOf

def plotAndPrepareMainFigures(resultsFolder="Results/MainFigures/", figuresToDo=["Fig. 2 B alternative", "Fig. 3 B alternative"]):
    Path(resultsFolder).mkdir(parents=True, exist_ok=True)
    # Fig. 2 A - .csv-file to use in MGX showing TP&TN in blue and FN&FP
    if figuresToDo == "all" or "Fig. 2 A" in figuresToDo:
        fig2ResultFolder = resultsFolder + "Fig 2/"
        Path(fig2ResultFolder).mkdir(parents=True, exist_ok=True)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=True,
                    featureSetName="allTopos", saveUnderFolder=fig2ResultFolder)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=True,
                    featureSetName="area", saveUnderFolder=fig2ResultFolder)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=True,
                    featureSetName="topoAndBio", saveUnderFolder=fig2ResultFolder)
    # Fig. 2 B - div prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 2 B" in figuresToDo:
        fig2ResultFolder = resultsFolder + "Fig 2/"
        Path(fig2ResultFolder).mkdir(parents=True, exist_ok=True)
        mainDivPredRandomization(performance="Acc", doMainFig=True,
                                 baseResultsFolder="Results/divEventData/manualCentres/",
                                 addOtherTestWithBaseFolder="Results/ktnDivEventData/manualCentres/",
                                 savePlotFolder=fig2ResultFolder)
    # Fig. 2 B alternative - div prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 2 B alternative" in figuresToDo:
        fontSize = 24
        fig2ResultFolder = resultsFolder + "Fig 2 alternative/"
        Path(fig2ResultFolder).mkdir(parents=True, exist_ok=True)
        mainDivPredRandomization(performance="Acc", doMainFig=True,
                                 baseResultsFolder="Results/divEventData/manualCentres/",
                                 resultsTestFilename=None,
                                 savePlotFolder=fig2ResultFolder,
                                 fontSize=fontSize)
        mainDivPredTestComparisons(savePlotFolder=fig2ResultFolder, fontSize=fontSize)
    # Fig. 3 A - .csv-file of correct and wrong topo predictions on example cell
    if figuresToDo == "all" or "Fig. 3 A" in figuresToDo:
        fig3AResultFolder = resultsFolder + "Fig 3/single cell vis/"
        Path(fig3AResultFolder).mkdir(parents=True, exist_ok=True)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=False,
                    featureSetName="allTopos", saveUnderFolder=fig3AResultFolder)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=False,
                    featureSetName="bio", saveUnderFolder=fig3AResultFolder)
        mainCreateTissuePredictionColoringOf(doDivPredVisualisation=False,
                    featureSetName="topoAndBio", saveUnderFolder=fig3AResultFolder)
    # Fig. 3 B - topo prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 3 B" in figuresToDo:
        fig3ResultFolder = resultsFolder + "Fig 3/"
        Path(fig3ResultFolder).mkdir(parents=True, exist_ok=True)
        mainTopoPredRandomization(performance="Acc", doMainFig=True,
                                 excludeDivNeighbours=True,
                                 baseResultsFolder="Results/topoPredData/diff/manualCentres/",
                                 addOtherTestWithBaseFolder="Results/ktnTopoPredData/diff/manualCentres/",
                                 savePlotFolder=fig3ResultFolder)
    # Fig. 3 B - topo prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 3 B alternative" in figuresToDo:
        fontSize = 24
        fig3ResultFolder = resultsFolder + "Fig 3 alternative/"
        Path(fig3ResultFolder).mkdir(parents=True, exist_ok=True)
        mainTopoPredRandomization(performance="Acc", doMainFig=True,
                                 excludeDivNeighbours=True,
                                 baseResultsFolder="Results/topoPredData/diff/manualCentres/",
                                 resultsTestFilename=None,
                                 savePlotFolder=fig3ResultFolder,
                                 fontSize=fontSize)
        mainTopoPredTestComparisons(savePlotFolder=fig3ResultFolder, fontSize=fontSize)
    # Fig. 3 C - topo prediction AUC results of different classes for WT test
    # Sup. Fig. 6 A-C - topo prediction ROC curves results of different classes for WT test
    if figuresToDo == "all" or "Fig. 3 C WT" in figuresToDo:
        fig3ResultFolder = resultsFolder + "Fig 3/AUC details WT/"
        Path(fig3ResultFolder).mkdir(parents=True, exist_ok=True)
        mainPlotRocCurvesAndAUCLabelDetails(resultsBaseFolder="Results/topoPredData/diff/manualCentres/",
                                            saveUnderFolder=fig3ResultFolder)
    # Fig. 3 C - topo prediction AUC results of different classes for ktn test
    # Sup. Fig. 6 E-G - topo prediction ROC curves results of different classes for ktn test
    if figuresToDo == "all" or "Fig. 3 C ktn" in figuresToDo:
        fig3ResultFolder = resultsFolder + "Fig 3/AUC details ktn/"
        Path(fig3ResultFolder).mkdir(parents=True, exist_ok=True)
        mainPlotRocCurvesAndAUCLabelDetails(resultsBaseFolder="Results/ktnTopoPredData/diff/manualCentres/",
                                            modelBaseFolder="Results/topoPredData/diff/manualCentres/",
                                            saveUnderFolder=fig3ResultFolder)
    # Fig. 4 B - example of predicted and observed topo prediction
    # Fig. 4 C - percentage of of correctly estimated neighbours per local topology
    # Fig. 4 D - concordance between the observed and predicted topologies
    # Fig. 4 E - comparison of obs. vs  pred. harmonic centrality applying div. and topo. prediction models

def plotAndPrepareSuppFigures(resultsFolder="Results/SuppFigures/", figuresToDo=["Fig. 9 B"]):
    Path(resultsFolder).mkdir(parents=True, exist_ok=True)
    # Fig. 2 - Pearson corr coeff of topological features vs area
    if figuresToDo == "all" or "Fig. 2" in figuresToDo:
        mainSaveCorrelation(baseFeatureFolder="Data/WT/divEventData/manualCentres/", savePlotFolder=resultsFolder)
    # Fig. 5 A - div prediction acc. randomisation results to further add text in power point
    # Fig. 5 B - topo prediction acc. randomisation results to further add text in power point
    # Fig. 8 - comparison of obs. vs  pred. of all centrality applying div. and topo. prediction models
    # Fig. 9 - density distributions of features - A topological and B biological features
    if figuresToDo == "all" or "Fig. 9 A" in figuresToDo:
        fig9AResultFolder = resultsFolder + "Fig 9A/"
        Path(fig9AResultFolder).mkdir(parents=True, exist_ok=True)
        mainSaveDensityPlotsOfFeaturesFromDiffScenarios(plotTopoFeatures=True, savePlotFolder=fig9AResultFolder)
    if figuresToDo == "all" or "Fig. 9 B" in figuresToDo:
        fig9BResultFolder = resultsFolder + "Fig 9B/"
        Path(fig9BResultFolder).mkdir(parents=True, exist_ok=True)
        mainSaveDensityPlotsOfFeaturesFromDiffScenarios(plotTopoFeatures=False, savePlotFolder=fig9BResultFolder)


if __name__== "__main__":
    plotAndPrepareMainFigures()
    # plotAndPrepareSuppFigures()
