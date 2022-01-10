import sys

sys.path.insert(0, "./Code/Predicting/")
from PredictonManager import PredictonManager

def mainTestKtn(runDivEventPred=True):
    dataFolder = "Data/ktn/"
    featureAndLabelFolder = "Data/ktn/topoPredData/"
    folderToSaveVal = None
    timePointsPerPlant = 3
    plantNames = ["ktnP1", "ktnP2", "ktnP3"]
    testPlants = ["ktnP1", "ktnP2", "ktnP3"]
    modelType =  {"modelType":"svm","kernel":"rbf"}
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
    allModelTypes = [{"modelType":"svm","kernel":"rbf"}, {"modelType":"random forest"}, {"modelType":"svm","kernel":"sigmoid"}]
    centralCellsDict = {"ktnP1": [ [], [3839, 3959] ],
                        "ktnP2": [ [23], [424, 426, 50] ],
                        "ktnP3": [ [29, 199,527], [424, 28, 431] ] }
    modelType = {"modelType":"svm", "kernel":"rbf"}
    doHyperParameterisation = True
    normalisePerTissue = False
    normaliseTrainTestData = True
    normaliseTrainValTestData = False
    featureProperty = "combinedTable"
    parName = None
    if runDivEventPred:
        for set in ["topoAndBio", "allTopos", "area", "topology", "lowCor0.3", "lowCor0.5", "lowCor0.7"]:
            print("testing division prediction with set ", set)
            labelName = "combinedLabels.csv"
            setFeatureAndLabelFolder = "{}divEventData/manualCentres/{}/".format(dataFolder, set)
            resultsFolder = "Results/ktnDivEventData/temp/{}/".format(set)
            useSpecificTestModelFilename = "Results/divEventData/temp/{}/svm_k2h_combinedTable_l3f0n1c0bal0ex0/testModel.pkl".format(set)
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
        sys.exit()
    for set in ["topoAndBio", "allTopos", "bio", "topology", "lowCor0.3", "lowCor0.5", "lowCor0.7"]:
        print("testing topo prediction with set ", set)
        labelName = "combinedLabels.csv"
        setFeatureAndLabelFolder = "{}topoPredData/diff/manualCentres/{}/".format(dataFolder, set)
        resultsFolder = "Results/ktnTopoPredData/diff/manualCentres/{}/".format(set)
        useSpecificTestModelFilename = "Results/topoPredData/diff/manualCentres/{}/svm_k2h_combinedTable_l3f0n1c0bal0ex1/testModel.pkl".format(set)
        if useTemporaryResultsFolder:
            resultsFolder = "Temporary/{}/".format(set)
        newResultsFolder = resultsFolder
        folderToSaveVal = newResultsFolder
        givenFeatureName = "combinedFeatures_{}_notnormalised.csv".format(set)
        print("newResultsFolder: " + newResultsFolder)
        modelType = {"modelType":"svm", "kernel":"rbf"}
        manager = PredictonManager(plantNames=plantNames,
                                   testPlants=testPlants,
                                   featureProperty=featureProperty,
                                   dataFolder=dataFolder,
                                   timePointsPerPlant=timePointsPerPlant,
                                   excludeDividingNeighbours=True, # in case you change excludeDividingNeighbours to False change ex1 to ex0 in the useSpecificTestModelFilename
                                   featureAndLabelFolder=featureAndLabelFolder,
                                   givenFeatureName=givenFeatureName,
                                   resultsFolder=newResultsFolder,
                                   nSplits=nSplits,
                                   balanceData=balanceData,
                                   modelType=modelType,
                                   runModelTraining = runModelTraining,
                                   runModelTesting = runModelTesting,
                                   useSpecificTestModelFilename=useSpecificTestModelFilename,
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
    mainTestKtn(runDivEventPred=True)
    mainTestKtn(runDivEventPred=False)
