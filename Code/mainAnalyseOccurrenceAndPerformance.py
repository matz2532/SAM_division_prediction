import sys

sys.path.insert(0, "./Code/Analysis Tools/")
sys.path.insert(0, "./Code/Propagation/")
from BarPlotPlotter import mainDivPredRandomization, mainTopoPredRandomization
from BarPlotPlotterExtended import mainDivPredTestComparisons, mainTopoPredTestComparisons
from CorrelationHeatMapDisplayer import mainSaveCorrelation
from DetailedCorrelationDataPlotter import compareModelPredictedVsRandomPropagationFeatureCorrelations, plotFeaturesCorrelationOfPredVsObsPropagation
from FeatureDensityPlotting import mainSaveDensityPlotsOfFeaturesFromDiffScenarios
from GeneralDataAnalyser import GeneralDataAnalyser
from LocalTopologyPredictionComparer import plotPercentageCorrectTopologies
from MyScorer import mainPlotRocCurvesAndAUCLabelDetails
from pathlib import Path
from VisualisePredictionsOnTissue import mainCreateTissuePredictionColoringOf

def plotAndPrepareMainFigures(resultsFolder="Results/MainFigures/", figuresToDo=["Fig. 4 D"]):
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
                                            featureSets=["topoAndBio", "lowCor0.3", "topology"],
                                            saveUnderFolder=fig3ResultFolder)
    # Fig. 4 B - example of predicted and observed topo prediction
    if figuresToDo == "all" or "Fig. 4 B" in figuresToDo:
        fig4ResultFolder = resultsFolder + "Fig 4/"
        Path(fig4ResultFolder).mkdir(parents=True, exist_ok=True)
    # Fig. 4 C - percentage of correctly estimated neighbours per local topology
    if figuresToDo == "all" or "Fig. 4 C" in figuresToDo:
        fig4ResultFolder = resultsFolder + "Fig 4/"
        Path(fig4ResultFolder).mkdir(parents=True, exist_ok=True)
        plotPercentageCorrectTopologies(baseResultsFolder="Results/DivAndTopoApplication/",
                                        folderToSave=fig4ResultFolder,
                                        plotName="Fig 4C - topology prediction density plot.png",
                                        fontSize=20)
    # Fig. 4 D - concordance between the observed and predicted topologies
    if figuresToDo == "all" or "Fig. 4 D" in figuresToDo:
        fig4ResultFolder = resultsFolder + "Fig 4/"
        Path(fig4ResultFolder).mkdir(parents=True, exist_ok=True)
        compareModelPredictedVsRandomPropagationFeatureCorrelations(saveUnderFolder=fig4ResultFolder)
    # Fig. 4 E - comparison of obs. vs pred. best correlating topological feature applying div. and topo. prediction models
    # Sup. Fig. 8 - comparison of obs. vs pred. non-best correlating topological feature applying div. and topo. prediction models
    if figuresToDo == "all" or "Fig. 4 E" in figuresToDo:
        fig4ResultFolder = resultsFolder + "Fig 4/"
        Path(fig4ResultFolder).mkdir(parents=True, exist_ok=True)
        plotFeaturesCorrelationOfPredVsObsPropagation(saveUnderFolder=fig4ResultFolder)

def plotAndPrepareSuppFigures(resultsFolder="Results/SuppFigures/", figuresToDo=["Fig. 5 A", "Fig. 5 B"]):
    Path(resultsFolder).mkdir(parents=True, exist_ok=True)
    # Fig. 2 - Pearson corr coeff of topological features vs area
    if figuresToDo == "all" or "Fig. 2" in figuresToDo:
        mainSaveCorrelation(baseFeatureFolder="Data/WT/divEventData/manualCentres/", savePlotFolder=resultsFolder)
    # Fig. 3 A - div prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 3 A" in figuresToDo:
        fontSize = 24
        fig2ResultFolder = resultsFolder + "Fig 3/"
        Path(fig2ResultFolder).mkdir(parents=True, exist_ok=True)
        mainDivPredRandomization(performance="AUC", doMainFig=True,
                                 baseResultsFolder="Results/divEventData/manualCentres/",
                                 addOtherTestWithBaseFolder="Results/ktnDivEventData/manualCentres/",
                                 savePlotFolder=fig2ResultFolder,
                                 fontSize=fontSize)
    # Fig. 3 B - div prediction acc. results to further add text in power point
    if figuresToDo == "all" or "Fig. 3 B" in figuresToDo:
        fontSize = 24
        fig2ResultFolder = resultsFolder + "Fig 3/"
        Path(fig2ResultFolder).mkdir(parents=True, exist_ok=True)
        mainTopoPredRandomization(performance="AUC", doMainFig=True,
                                  baseResultsFolder="Results/topoPredData/diff/manualCentres/",
                                  addOtherTestWithBaseFolder="Results/ktnTopoPredData/diff/manualCentres/",
                                  savePlotFolder=fig2ResultFolder,
                                  fontSize=fontSize)
    # Fig. 5 A - div prediction acc. randomisation results to further add text in power point
    if figuresToDo == "all" or "Fig. 5 A" in figuresToDo:
        fontSize = 24
        fig5AResultFolder = resultsFolder + "Fig 5/div pred random/"
        Path(fig5AResultFolder).mkdir(parents=True, exist_ok=True)
        mainDivPredRandomization(performance="Acc", plotOnlyRandom=True,
                                 baseResultsFolder="Results/divEventData/manualCentres/",
                                 savePlotFolder=fig5AResultFolder,
                                 fontSize=fontSize)
    # Fig. 5 B - topo prediction acc. randomisation results to further add text in power point
    if figuresToDo == "all" or "Fig. 5 B" in figuresToDo:
        fontSize = 24
        fig5BResultFolder = resultsFolder + "Fig 5/topo pred random/"
        Path(fig5BResultFolder).mkdir(parents=True, exist_ok=True)
        mainTopoPredRandomization(performance="Acc", plotOnlyRandom=True,
                                 baseResultsFolder="Results/topoPredData/diff/manualCentres/",
                                 savePlotFolder=fig5BResultFolder,
                                 fontSize=fontSize)
    # Fig. 8 - comparison of obs. vs  pred. of all centrality applying div. and topo. prediction models
    # see Main Fig. 4 E
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
