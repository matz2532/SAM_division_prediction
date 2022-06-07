import pandas as pd
import sys

sys.path.insert(0, "./Code/Predicting/")
from PredictonManager import PredictonManager

def mainTestKtn(runDivEventPred=True, givenSets=None, excludeDividingNeighbours=True):
    dataFolder = "Data/ktn/"
    featureAndLabelFolder = "Data/ktn/topoPredData/"
    folderToSaveVal = None
    timePointsPerPlant = 3
    plantNames = ["ktnP1", "ktnP2", "ktnP3"]
    testPlants = ["ktnP1", "ktnP2", "ktnP3"]
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
    centralCellsDict = {"ktnP1": [ [], [3839, 3959] ],
                        "ktnP2": [ [23], [424, 426, 50] ],
                        "ktnP3": [ [29, 199,527], [424, 28, 431] ] }
    modelType = {"modelType":"svm", "kernel":"linear"}
    doHyperParameterisation = True
    normalisePerTissue = False
    normaliseTrainTestData = True
    normaliseTrainValTestData = False
    featureProperty = "combinedTable"
    parName = None
    if runDivEventPred:
        if givenSets is None:
            givenSets = ["allTopos", "area", "topoAndBio", "topology", "lowCor0.3"]
        for set in givenSets:
            print("testing division prediction with set ", set)
            labelName = "combinedLabels.csv"
            setFeatureAndLabelFolder = "{}divEventData/manualCentres/{}/".format(dataFolder, set)
            resultsFolder = "Results/ktnDivEventData/manualCentres/{}/".format(set)
            previousModelFolder = "Results/divEventData/manualCentres/{}/svm_k1h_combinedTable_l3f0n1c0bal0ex0/".format(set)
            useSpecificTestModelFilename = previousModelFolder + "testModel.pkl"
            useGivenFeatureColumns = list(pd.read_csv(previousModelFolder + "normalizedFeatures_train.csv").columns)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("newResultsFolder: " + newResultsFolder)
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
        if givenSets is None:
            givenSets = ["allTopos", "bio", "topoAndBio", "topology", "lowCor0.3"]
        for set in givenSets:
            print("testing topo prediction with set ", set)
            labelName = "combinedLabels.csv"
            setFeatureAndLabelFolder = "{}topoPredData/diff/manualCentres/{}/".format(dataFolder, set)
            resultsFolder = "Results/ktnTopoPredData/diff/manualCentres/{}/".format(set)
            if excludeDividingNeighbours:
                excludeText = "ex1"
            else:
                excludeText = "ex0"
            previousModelFolder = "Results/topoPredData/diff/manualCentres/{}/svm_k1h_combinedTable_l3f0n1c0bal0/".format(set, excludeText)
            useSpecificTestModelFilename = previousModelFolder + "testModel.pkl"
            useGivenFeatureColumns = list(pd.read_csv(previousModelFolder + "normalizedFeatures_train.csv").columns)
            if useTemporaryResultsFolder:
                resultsFolder = "Temporary/{}/".format(set)
            newResultsFolder = resultsFolder
            folderToSaveVal = newResultsFolder
            givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
            print("newResultsFolder: " + newResultsFolder)
            manager = PredictonManager(plantNames=plantNames,
                                       testPlants=testPlants,
                                       featureProperty=featureProperty,
                                       dataFolder=dataFolder,
                                       timePointsPerPlant=timePointsPerPlant,
                                       excludeDividingNeighbours=excludeDividingNeighbours, # in case you change excludeDividingNeighbours to False change ex1 to ex0 in the useSpecificTestModelFilename
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

if __name__ == '__main__':
    # mainTestKtn(runDivEventPred=True, givenSets=None)
    from mainTestFloralMeristems import mainTestFloralMeristems
    mainTestKtn(runDivEventPred=False, givenSets=None, excludeDividingNeighbours=False)
    mainTestFloralMeristems(runDivEventPred=False, useWT=True, excludeDividingNeighbours=False)
    mainTestFloralMeristems(runDivEventPred=False, useWT=False, excludeDividingNeighbours=False)
