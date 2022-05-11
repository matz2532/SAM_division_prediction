import sys

sys.path.insert(0, "./Code/Predicting/")
from PredictonManager import PredictonManager

def mainTrainValTestWT(runDivEventPred=True, givenSets=None, baseResultsFolder=None,
                       excludeDividingNeighboursProperties=[True, False],
                       runModelTraining=True, runModelTesting=False,
                       saveLearningCurve=False, modelType={"modelType":"svm","kernel":"rbf"}):
    dataFolder = "Data/WT/"
    plantNames = ["P1", "P2", "P5", "P6", "P8", "P9", "P10", "P11"]
    testPlants = ["P2", "P9"]
    usePreviousTrainedModelsIfPossible = False
    onlyTestModelWithoutTrainingData = False
    useManualCentres = True
    # print options:
    printBalancedLabelCount = True
    nSplits = "per plant"
    balanceData = False
    hyperParameters = None
    modelNameExtension = ""
    centralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [], [], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [], [], [358, 189]],
                        "P9":[[1047, 721, 1048], [7303, 7533], [6735, 7129], [2160, 2228], [7366, 7236]],
                        "P10":[[1511, 1524], [7281, 7516, 7534], [], [7634, 7722, 7795, 7794], [1073, 1074, 892]],
                        "P11":[[1751], [9489], [9759, 9793], [3300, 3211, 3060], [3956, 3979]]}
    doHyperParameterisation = True
    normalisePerTissue = False
    normaliseTrainTestData = True
    normaliseTrainValTestData = False
    featureProperty = "combinedTable"
    if runDivEventPred:
        if givenSets is None:
            givenSets = ["allTopos", "area", "topoAndBio", "topology", "lowCor0.3", "lowCor0.5", "lowCor0.7"]
        if baseResultsFolder is None:
            baseResultsFolder = "Results/divEventData/"
        for set in givenSets:
            labelName = "combinedLabels.csv"
            if useManualCentres:
                setFeatureAndLabelFolder = "Data/WT/divEventData/manualCentres/{}/".format(set)
                resultsFolder = "{}manualCentres/{}/".format(baseResultsFolder, set)
            else:
                setFeatureAndLabelFolder = "Data/WT/divEventData/{}/".format(set)
                resultsFolder = "{}{}/".format(baseResultsFolder, set)
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("resultsFolder: "+resultsFolder)
            manager = PredictonManager(plantNames=plantNames,
                                   testPlants=testPlants,
                                   featureProperty=featureProperty,
                                   dataFolder=dataFolder,
                                   featureAndLabelFolder=setFeatureAndLabelFolder,
                                   givenFeatureName=givenFeatureName,
                                   resultsFolder=resultsFolder,
                                   nSplits=nSplits,
                                   balanceData=balanceData,
                                   modelType=modelType,
                                   usePreviousTrainedModelsIfPossible=usePreviousTrainedModelsIfPossible,
                                   runModelTraining=runModelTraining,
                                   runModelTesting = runModelTesting,
                                   onlyTestModelWithoutTrainingData=onlyTestModelWithoutTrainingData,
                                   saveLearningCurve=saveLearningCurve,
                                   centralCellsDict=centralCellsDict,
                                   normalisePerTissue=normalisePerTissue,
                                   normaliseTrainTestData=normaliseTrainTestData,
                                   normaliseTrainValTestData=normaliseTrainValTestData,
                                   doHyperParameterisation=doHyperParameterisation,
                                   hyperParameters=hyperParameters,
                                   modelNameExtension=modelNameExtension,
                                   folderToSaveVal=resultsFolder,
                                   setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                   useOnlyTwo=True,
                                   labelName=labelName,
                                   excludeDividingNeighbours=False,
                                   printBalancedLabelCount=printBalancedLabelCount)
    else:
        if givenSets is None:
            givenSets = ["allTopos", "bio", "topoAndBio", "topology", "lowCor0.3", "lowCor0.5", "lowCor0.7"]
        if baseResultsFolder is None:
            baseResultsFolder = "Results/topoPredData/"
        for excludeDividingNeighbours in excludeDividingNeighboursProperties:
                for set in givenSets:
                    labelName = "combinedLabels.csv"
                    if useManualCentres:
                        setFeatureAndLabelFolder = "Data/WT/topoPredData/diff/manualCentres/{}/".format(set)
                        resultsFolder = "{}diff/manualCentres/{}/".format(baseResultsFolder, set)
                    else:
                        setFeatureAndLabelFolder = "Data/WT/topoPredData/diff/{}/".format(set)
                        resultsFolder = "{}diff/{}/".format(baseResultsFolder, set)
                    givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
                    print("resultsFolder: "+resultsFolder)
                    manager = PredictonManager(plantNames=plantNames,
                                               testPlants=testPlants,
                                               featureProperty=featureProperty,
                                               dataFolder=dataFolder,
                                               featureAndLabelFolder=setFeatureAndLabelFolder,
                                               givenFeatureName=givenFeatureName,
                                               resultsFolder=resultsFolder,
                                               nSplits=nSplits,
                                               balanceData=balanceData,
                                               modelType=modelType,
                                               usePreviousTrainedModelsIfPossible=usePreviousTrainedModelsIfPossible,
                                               runModelTraining = runModelTraining,
                                               runModelTesting = runModelTesting,
                                               saveLearningCurve = saveLearningCurve,
                                               plotLearningCurveLegend=True if set != "bio" else False,
                                               centralCellsDict=centralCellsDict,
                                               normalisePerTissue=normalisePerTissue,
                                               normaliseTrainTestData=normaliseTrainTestData,
                                               normaliseTrainValTestData=normaliseTrainValTestData,
                                               doHyperParameterisation=doHyperParameterisation,
                                               hyperParameters=hyperParameters,
                                               modelNameExtension=modelNameExtension,
                                               folderToSaveVal=resultsFolder,
                                               setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                               labelName=labelName,
                                               excludeDividingNeighbours=excludeDividingNeighbours,
                                               printBalancedLabelCount=printBalancedLabelCount)


if __name__ == '__main__':
    mainTrainValTestWT(runDivEventPred=True, runModelTraining=True, saveLearningCurve=False,
                      runModelTesting=False, modelType={"modelType":"svm","kernel":"linear"})
    mainTrainValTestWT(runDivEventPred=True, runModelTraining=True, saveLearningCurve=False,
                       runModelTesting=False,givenSets = ["allTopos","topoAndBio", "lowCor0.3", "lowCor0.5", "lowCor0.7"])
    mainTrainValTestWT(runDivEventPred=False, runModelTraining=True, saveLearningCurve=False,
                      excludeDividingNeighboursProperties=[True], runModelTesting=False,
                      givenSets = ["allTopos", "topoAndBio", "lowCor0.3", "lowCor0.5", "lowCor0.7"])
    mainTrainValTestWT(runDivEventPred=False, runModelTraining=True, saveLearningCurve=False,
                       excludeDividingNeighboursProperties=[True], runModelTesting=False, modelType={"modelType":"svm","kernel":"linear"})
