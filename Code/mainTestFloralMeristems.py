import pandas as pd
import sys

sys.path.insert(0, "./Code/Predicting/")
from PredictonManager import PredictonManager

def mainTestFloralMeristems(runDivEventPred=True, useWT=True, givenSets=None, excludeDividingNeighbours=True):
    folderToSaveVal = None
    modelType =  {"modelType":"svm", "kernel":"linear"}
    useTemporaryResultsFolder = False
    runModelTraining = False
    runModelTesting = True
    useManualCentres = True
    # print options:
    printBalancedLabelCount = True
    nSplits = "per plant"
    balanceData = False
    parametersToAddOrOverwrite = None
    hyperParameters = None
    modelNameExtension = ""
    doHyperParameterisation = True
    normalisePerTissue = False
    normaliseTrainTestData = True
    normaliseTrainValTestData = False
    featureProperty = "combinedTable"
    parName = None
    dataBaseFolder = "Data/"
    if useWT:
        centralCellsDict = {"p4 FM1" : [[21523], [21962, 21944], [22124, 22343, 22329], [22970, 22645, 22961], [24041, 24232]],
                            "p4 FM2" : [[164], [29886, 29856], [30336], [31576, 31129, 30829], [32691, 32822, 32823 ]]}
        dataExtension = "floral meristems/WT/"
        timePointsPerPlant = 5
    else:
        centralCellsDict = {"p3 FM6 flower 1" : [[13707, 13331], [12893, 12892], [14326, 14335, 14079], [], []],
                            "p3 FM6 flower 2" : [[43790, 44025], [43596, 43597], [19672, 19286], [], []],
                            "p3 FM6 flower 3" : [[], [20610], [20654, 20652, 20716], [21902, 22115, 22567], [40529, 39982]]}
        dataExtension = "floral meristems/ktn/"
        timePointsPerPlant = 5
    if runDivEventPred:
        predictionTypeExtension = "divEventData/"
        if givenSets is None:
            givenSets = ["allTopos", "area", "topoAndBio", "topology", "lowCor0.3"]
    else:
        predictionTypeExtension = "topoPredData/"
        if givenSets is None:
            givenSets = ["allTopos", "bio", "topoAndBio", "topology", "lowCor0.3"]
    dataFolder = dataBaseFolder + dataExtension
    featureAndLabelFolder = dataFolder + predictionTypeExtension
    plantNames = list(centralCellsDict.keys())
    testPlants = plantNames

    if runDivEventPred:
        for set in givenSets:
            print("testing division prediction with set ", set)
            labelName = "combinedLabels.csv"
            setFeatureAndLabelFolder = "{}manualCentres/{}/".format(featureAndLabelFolder, set)
            resultsFolder = "Results/{}manualCentres/{}/".format(dataExtension + predictionTypeExtension, set)
            previousModelFolder = "Results/divEventData/manualCentres/{}/svm_k1h_combinedTable_l3f0n1c0bal0ex0/".format(set)
            useSpecificTestModelFilename = previousModelFolder + "testModel.pkl"
            useGivenFeatureColumns = list(pd.read_csv(previousModelFolder + "normalizedFeatures_train.csv").columns)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            manager = PredictonManager(plantNames=plantNames,
                                   testPlants=testPlants,
                                   featureProperty=featureProperty,
                                   dataFolder=dataFolder,
                                   featureAndLabelFolder=featureAndLabelFolder,
                                   givenFeatureName=givenFeatureName,
                                   resultsFolder=newResultsFolder,
                                   nSplits=nSplits,
                                   balanceData=balanceData,
                                   modelType=modelType,
                                   runModelTraining=runModelTraining,
                                   runModelTesting = runModelTesting,
                                   useSpecificTestModelFilename=useSpecificTestModelFilename,
                                   useGivenFeatureColumns=useGivenFeatureColumns,
                                   normaliseOnTestData=True,
                                   centralCellsDict=centralCellsDict,
                                   normalisePerTissue=normalisePerTissue,
                                   normaliseTrainTestData=normaliseTrainTestData,
                                   normaliseTrainValTestData=normaliseTrainValTestData,
                                   doHyperParameterisation=doHyperParameterisation,
                                   hyperParameters=hyperParameters,
                                   parametersToAddOrOverwrite=parametersToAddOrOverwrite,
                                   specialParName=parName,
                                   modelNameExtension=modelNameExtension,
                                   folderToSaveVal=folderToSaveVal,
                                   setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                   useOnlyTwo=True,
                                   labelName=labelName,
                                   excludeDividingNeighbours=False,
                                   printBalancedLabelCount=printBalancedLabelCount)
    else:
        for set in givenSets:
            print("testing topo prediction with set ", set)
            labelName = "combinedLabels.csv"
            setFeatureAndLabelFolder = "{}diff/manualCentres/{}/".format(featureAndLabelFolder, set)
            resultsFolder = "Results/{}diff/manualCentres/{}/".format(dataExtension + predictionTypeExtension, set)
            if excludeDividingNeighbours:
                excludeText = "ex1"
            else:
                excludeText = "ex0"
            previousModelFolder = "Results/topoPredData/diff/manualCentres/{}/svm_k1h_combinedTable_l3f0n1c0bal0{}/".format(set, excludeText)
            useSpecificTestModelFilename = previousModelFolder + "testModel.pkl"
            useGivenFeatureColumns = list(pd.read_csv(previousModelFolder + "normalizedFeatures_train.csv").columns)
            if useTemporaryResultsFolder:
                resultsFolder = "Temporary/{}/".format(set)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            manager = PredictonManager(plantNames=plantNames,
                                       testPlants=testPlants,
                                       featureProperty=featureProperty,
                                       dataFolder=dataFolder,
                                       timePointsPerPlant=timePointsPerPlant,
                                       excludeDividingNeighbours=excludeDividingNeighbours,
                                       featureAndLabelFolder=featureAndLabelFolder,
                                       givenFeatureName=givenFeatureName,
                                       resultsFolder=newResultsFolder,
                                       nSplits=nSplits,
                                       balanceData=balanceData,
                                       modelType=modelType,
                                       runModelTraining = runModelTraining,
                                       runModelTesting = runModelTesting,
                                       useSpecificTestModelFilename=useSpecificTestModelFilename,
                                       useGivenFeatureColumns=useGivenFeatureColumns,
                                       normaliseOnTestData=True,
                                       centralCellsDict=centralCellsDict,
                                       normalisePerTissue=normalisePerTissue,
                                       normaliseTrainTestData=normaliseTrainTestData,
                                       normaliseTrainValTestData=normaliseTrainValTestData,
                                       doHyperParameterisation=doHyperParameterisation,
                                       hyperParameters=hyperParameters,
                                       parametersToAddOrOverwrite=parametersToAddOrOverwrite,
                                       specialParName=parName,
                                       modelNameExtension=modelNameExtension,
                                       folderToSaveVal=folderToSaveVal,
                                       setFeatureAndLabelFolder=setFeatureAndLabelFolder,
                                       labelName=labelName,
                                       printBalancedLabelCount=printBalancedLabelCount)

if __name__== "__main__":
    mainTestFloralMeristems(runDivEventPred=True, useWT=True)
    mainTestFloralMeristems(runDivEventPred=True, useWT=False)
    mainTestFloralMeristems(runDivEventPred=False, useWT=True, excludeDividingNeighbours=True)
    mainTestFloralMeristems(runDivEventPred=False, useWT=False, excludeDividingNeighbours=True)
    mainTestFloralMeristems(runDivEventPred=False, useWT=True, excludeDividingNeighbours=False)
    mainTestFloralMeristems(runDivEventPred=False, useWT=False, excludeDividingNeighbours=False)
