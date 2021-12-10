import sys

sys.path.insert(0, "./Code/Predicting/")
from PredictonManager import PredictonManager

def main():
    dataFolder = "Data/WT/"
    plantNames = ["P1", "P2", "P5", "P6", "P8", "P9", "P10", "P11"]
    testPlants = ["P2", "P9"]
    modelType =  {"modelType":"svm","kernel":"rbf"}
    runDivEventPred = True
    usePreviousTrainedModelsIfPossible = True
    runModelTraining = True
    runModelTesting = False
    onlyTestModelWithoutTrainingData = False
    saveLearningCurve = False
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
        for set in ["allTopos", "area", "topoAndBio", "topology", "lowCor0.3", "lowCor0.5", "lowCor0.7"]:
            labelName = "combinedLabels.csv"
            if useManualCentres:
                setFeatureAndLabelFolder = "Data/WT/divEventData/manualCentres/{}/".format(set)
                resultsFolder = "Results/divEventData/manualCentres/{}/".format(set)
            else:
                setFeatureAndLabelFolder = "Data/WT/divEventData/{}/".format(set)
                resultsFolder = "Results/divEventData/{}/".format(set)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("newResultsFolder: "+newResultsFolder)
            manager = PredictonManager(plantNames=plantNames,
                                   testPlants=testPlants,
                                   featureProperty=featureProperty,
                                   dataFolder=dataFolder,
                                   featureAndLabelFolder=setFeatureAndLabelFolder,
                                   givenFeatureName=givenFeatureName,
                                   resultsFolder=newResultsFolder,
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
                                   selectedData=selectedData,
                                   modelNameExtension=modelNameExtension,
                                   folderToSaveVal=folderToSaveVal,
                                   setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                   useOnlyTwo=True,
                                   labelName=labelName,
                                   excludeDividingNeighbours=False,
                                   printBalancedLabelCount=printBalancedLabelCount)
        sys.exit()
    for excludeDividingNeighbours in [True, False]:
        for set in ["allTopos", "bio", "topoAndBio", "topology", "lowCor0.3", "lowCor0.5", "lowCor0.7"]:
            labelName = "combinedLabels.csv"
            if useManualCentres:
                setFeatureAndLabelFolder = "Data/WT/topoPredData/diff/manualCentres/{}/".format(set)
                resultsFolder = "Results/topoPredData/diff/manualCentres/{}/".format(set)
            else:
                setFeatureAndLabelFolder = "Data/WT/topoPredData/diff/{}/".format(set)
                resultsFolder = "Results/topoPredData/diff/{}/".format(set)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("newResultsFolder: "+newResultsFolder)
            manager = PredictonManager(plantNames=plantNames,
                                       testPlants=testPlants,
                                       featureProperty=featureProperty,
                                       dataFolder=dataFolder,
                                       featureAndLabelFolder=setFeatureAndLabelFolder,
                                       givenFeatureName=givenFeatureName,
                                       resultsFolder=newResultsFolder,
                                       nSplits=nSplits,
                                       balanceData=balanceData,
                                       modelType=modelType,
                                       usePreviousTrainedModelsIfPossible=usePreviousTrainedModelsIfPossible,
                                       runModelTraining = runModelTraining,
                                       runModelTesting = runModelTesting,
                                       saveLearningCurve = saveLearningCurve,
                                       centralCellsDict=centralCellsDict,
                                       normalisePerTissue=normalisePerTissue,
                                       normaliseTrainTestData=normaliseTrainTestData,
                                       normaliseTrainValTestData=normaliseTrainValTestData,
                                       doHyperParameterisation=doHyperParameterisation,
                                       hyperParameters=hyperParameters,
                                       selectedData=selectedData,
                                       modelNameExtension=modelNameExtension,
                                       folderToSaveVal=folderToSaveVal,
                                       setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                       labelName=labelName,
                                       excludeDividingNeighbours=excludeDividingNeighbours,
                                       printBalancedLabelCount=printBalancedLabelCount)

if __name__ == '__main__':
    main()
