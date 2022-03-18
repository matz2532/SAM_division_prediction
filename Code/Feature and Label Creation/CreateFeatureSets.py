import numpy as np
import pandas as pd
import scipy.stats
import shutil, sys

sys.path.insert(0, "./Code/DivEventPrediction/")
sys.path.insert(0, "./Code/Feature and Label Creation/")
sys.path.insert(0, "./Code/Predicting/")

from BiologicalFeatureCreatorForNetworkRecreation import BiologicalFeatureCreatorForNetworkRecreation
from DivEventDataCreator import DivEventDataCreator
from pathlib import Path
from TopologyPredictonDataCreator import TopologyPredictonDataCreator

class CreateFeatureSets (object):

    def __init__(self, dataFolder, folderToSave, setRange=None,
                 useManualCentres=True, estimateFeatures=True, estimateLabels=True,
                 plantNames=["P1", "P2", "P5", "P6", "P8"], useTopoCreator=False,
                 timePointsPerPlant=5,
                 centralCellsDict=None, skipEmptyCentrals=False, set=None,
                 takeCorrelationFromDifferentFolder=None, keepFromFolder=None):
        self.dataFolder = dataFolder
        self.folderToSave = folderToSave
        self.plantNames = plantNames
        self.estimateFeatures = estimateFeatures
        self.estimateLabels = estimateLabels
        self.useManualCentres = useManualCentres
        self.timePointsPerPlant = timePointsPerPlant
        self.centralCellsDict = centralCellsDict
        self.skipEmptyCentrals = skipEmptyCentrals
        self.takeCorrelationFromDifferentFolder = takeCorrelationFromDifferentFolder
        self.keepFromFolder = keepFromFolder
        if setRange is None:
            setRange = np.arange(8)
        for set in setRange:
            print("set: ", set)
            self.saveFeatureSets(set=set, dataFolder=dataFolder, plantNames=plantNames,
                            folderToSave=folderToSave,
                            estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                            useManualCentres=useManualCentres,
                            useTopoCreator=useTopoCreator,
                            centralCellsDict=self.centralCellsDict,
                            takeCorrelationFromDifferentFolder=self.takeCorrelationFromDifferentFolder,
                            keepFromFolder=self.keepFromFolder)

    def saveFeatureSets(self, set, dataFolder, plantNames, folderToSave,
                    estimateFeatures=True, estimateLabels=True,
                    useManualCentres=False, useTopoCreator=False,
                    centralCellsDict=None, takeCorrelationFromDifferentFolder=None,
                    keepFromFolder=None):
        specialGraphProperties = None
        featureProperty = "topology"
        onlyAra = False
        if useManualCentres:
            folderToSave += "manualCentres/"
            if centralCellsDict is None:
                centralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]], # in T2 5380 was not found check tissue in MGX
                                    "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                                    "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                                    "P6":[[861], [1651, 1621], [1763, 1844], [2109, 2176], [2381]],
                                    "P8":[[3241, 2869, 3044], [3421, 3657], [2805, 2814, 2876], [3013], [358, 189]]}
        else:
            centralCellsDict = None
        labelsFilenameToCopy = folderToSave+"{}/combinedLabels.csv".format(featureProperty)
        if set == 1:
            # for area
            featureProperty = "topologyArea"
            Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
            estimateLabels = False
            specialGraphProperties = {"useEdgeWeight":True,
                                      "invertEdgeWeight": True,
                                      "useSharedWallWeight": False,
                                      "useDistanceWeight": False,
                                      "maxNormEdgeWeightPerGraph": False}
        elif set == 2:
            # for Wall
            featureProperty = "topologyWall"
            Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
            estimateLabels = False
            specialGraphProperties = {"useEdgeWeight":True,
                                      "invertEdgeWeight": False,
                                      "useSharedWallWeight": True,
                                      "useDistanceWeight": False,
                                      "maxNormEdgeWeightPerGraph": False}
        elif set == 3:
            # for distance
            featureProperty = "topologyDist"
            Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
            estimateLabels = False
            specialGraphProperties = {"useEdgeWeight":True,
                                      "invertEdgeWeight": False,
                                      "useSharedWallWeight": False,
                                      "useDistanceWeight": True,
                                      "maxNormEdgeWeightPerGraph": False}
        elif set == 4:
            if useTopoCreator:
                featureProperty = "bio"
            else:
                featureProperty = "area"
            Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
            newLabelFilename = folderToSave+"{}/combinedLabels.csv".format(featureProperty)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, newLabelFilename)
            if useTopoCreator:
                labelTable = pd.read_csv(newLabelFilename)
                filenameToSave = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featureProperty, featureProperty)
                BiologicalFeatureCreatorForNetworkRecreation(baseFolder=dataFolder,
                        featureFilenameToSaveTo=filenameToSave,
                        oldFeatureTable=labelTable).CreateBiologicalFeatures()
                return None
            estimateLabels = False
            onlyAra = True
        elif set == 5:
            featureProperty = "allTopos"
            Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
            if useTopoCreator:
                self.combineTopoFeatures(folderToSave, featureProperty)
            else:
                self.combineFeatures(folderToSave, featureProperty)
            return None
        elif set == 6:
            featureProperty = "topoAndBio"
            Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
            if estimateLabels:
                shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
            if useTopoCreator:
                self.combineTopoFeatures(folderToSave, featureProperty, includeBioFeatures=True)
            else:
                self.combineFeatures(folderToSave, featureProperty, includeBioFeatures=True)
            return None
        elif set == 7:
            for rThreshold in [0.3, 0.5, 0.7]:
                featurePropertyToLoad = "allTopos"
                featureProperty = "lowCor{}".format(rThreshold)
                print(featureProperty)
                Path(folderToSave+"/"+featureProperty).mkdir(parents=True, exist_ok=True)
                if keepFromFolder:
                    self.saveBasedOnOhterCorrelationFolder(folderToSave, featureProperty,
                                                           featurePropertyToLoad, keepFromFolder)
                else:
                    if useTopoCreator:
                        self.saveLowCorrelationOf(folderToSave, featureProperty, featurePropertyToLoad,
                                rThreshold, splitOfFormatInfo=4, propertyCorAgainst="bio", corAgainstIdx=np.arange(4,10),
                                takeCorrelationFromDifferentFolder=takeCorrelationFromDifferentFolder)
                    else:
                        self.saveLowCorrelationOf(folderToSave, featureProperty, featurePropertyToLoad, rThreshold,
                                takeCorrelationFromDifferentFolder=takeCorrelationFromDifferentFolder)
                if estimateLabels:
                    shutil.copy2(labelsFilenameToCopy, folderToSave+"{}/combinedLabels.csv".format(featureProperty))
            return None
        if useTopoCreator:
            dataCreator = TopologyPredictonDataCreator(dataFolder, self.timePointsPerPlant,
                                                    plantNames,
                                                    specialGraphProperties=specialGraphProperties,
                                                    centralCellsDict=centralCellsDict)
            if set > 0:
                estimateLabels = folderToSave+"{}/combinedLabels.csv".format(featureProperty)
                if not Path(estimateLabels).is_file():
                    estimateLabels = True
        else:
            dataCreator = DivEventDataCreator(dataFolder, self.timePointsPerPlant,
                                              plantNames,
                                              specialGraphProperties=specialGraphProperties,
                                              centralCellsDict=centralCellsDict,
                                              onlyAra=onlyAra)
        dataCreator.MakeTrainingData(estimateFeatures=estimateFeatures, estimateLabels=estimateLabels,
                                     skipEmptyCentrals=self.skipEmptyCentrals)
        trainingData = dataCreator.GetFeatureTable()
        labelData = dataCreator.GetLabelTable()
        folderToSave += featureProperty + "/"
        Path(folderToSave).mkdir(parents=True, exist_ok=True)
        dataCreator.SaveFeatureTable(folderToSave+"combinedFeatures_{}_notnormalised.csv".format(featureProperty))
        dataCreator.SaveLabelTable(folderToSave+"combinedLabels.csv")

    def combineFeatures(self, folderToSave, featurePropertyName="", includeBioFeatures=False):
        allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist"]
        if includeBioFeatures:
            allFeatureProperties.insert(0, "area")
        allTables = []
        for i, featureProperty in enumerate(allFeatureProperties):
            filenameToLoad = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featureProperty, featureProperty)
            table = pd.read_csv(filenameToLoad)
            if i != 0:
                table = table.drop(columns=["plant", "time point", "cell"], axis=1)
            allTables.append(table)
        allTables = pd.concat(allTables, axis=1)
        if featurePropertyName != "":
            folderToSave += "{}/".format(featurePropertyName)
            filenameToSave = folderToSave + "combinedFeatures_{}_notnormalised.csv".format(featurePropertyName)
        else:
            filenameToSave = folderToSave + "combinedLabels_notnormalised.csv"
        Path(folderToSave).mkdir(parents=True, exist_ok=True)
        allTables.to_csv(filenameToSave, index=False)

    def combineTopoFeatures(self, folderToSave, featurePropertyName="", includeBioFeatures=False):
        allFeatureProperties = ["topology", "topologyArea", "topologyWall", "topologyDist"]
        if includeBioFeatures:
            allFeatureProperties.insert(0, "bio")
        allTableStarts = []
        allTableEnds = []
        for i, featureProperty in enumerate(allFeatureProperties):
            filenameToLoad = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featureProperty, featureProperty)
            table = pd.read_csv(filenameToLoad)
            nrOfFeatures = (table.shape[1] - 4)
            split = (nrOfFeatures // 2)
            if i != 0:
                columnsToDrop = np.asarray(table.columns)[:4]
                table = table.drop(columns=columnsToDrop, axis=1)
            else:
                split += 4
            if i == 0 and includeBioFeatures:
                allTableStarts.append(table)
            else:
                allTableStarts.append(table.iloc[:, :split])
                allTableEnds.append(table.iloc[:, split:])
        allTableStarts = pd.concat(allTableStarts, axis=1)
        allTableEnds = pd.concat(allTableEnds, axis=1)
        allTables = pd.concat([allTableStarts, allTableEnds], axis=1)
        if featurePropertyName != "":
            folderToSave += "{}/".format(featurePropertyName)
            filenameToSave = folderToSave + "combinedFeatures_{}_notnormalised.csv".format(featurePropertyName)
        else:
            filenameToSave = folderToSave + "combinedLabels_notnormalised.csv"
        Path(folderToSave).mkdir(parents=True, exist_ok=True)
        allTables.to_csv(filenameToSave, index=False)

    def saveLowCorrelationOf(self, folderToSave, featurePropertyToSave,
                             featurePropertyToLoad, rThreshold, propertyCorAgainst="area",
                             splitOfFormatInfo=3, corAgainstIdx=-1,
                             takeCorrelationFromDifferentFolder=None):
        filenameToLoad = folderToSave+"{}/combinedFeatures_{}_notnormalised.csv".format(featurePropertyToLoad, featurePropertyToLoad)
        table = pd.read_csv(filenameToLoad)
        standardFormat = table.iloc[:, :splitOfFormatInfo]
        if takeCorrelationFromDifferentFolder is None:
            valuesToCorrelate = table.iloc[:, splitOfFormatInfo:]
            folderToLoad = folderToSave
        else:
            folderToLoad = takeCorrelationFromDifferentFolder
            correlateOnDifferentTable = pd.read_csv(folderToLoad+"{}/combinedFeatures_{}_notnormalised.csv".format(featurePropertyToLoad, featurePropertyToLoad))
            valuesToCorrelate = correlateOnDifferentTable.iloc[:, splitOfFormatInfo:]
        filenameToLoad = folderToLoad+"{}/combinedFeatures_{}_notnormalised.csv".format(propertyCorAgainst, propertyCorAgainst)
        corAgainstValues = pd.read_csv(filenameToLoad).iloc[:, corAgainstIdx]
        columnsToKeep = self.columnsWithLowerCorrelation(valuesToCorrelate, corAgainstValues, rThreshold)
        reducedValues = table.iloc[:, splitOfFormatInfo:].iloc[:, columnsToKeep]
        reducedTable = pd.concat([standardFormat, reducedValues], axis=1)
        folderToSave = folderToSave + "{}/".format(featurePropertyToSave)
        Path(folderToSave).mkdir(parents=True, exist_ok=True)
        filenameToSave = folderToSave + "combinedFeatures_{}_notnormalised.csv".format(featurePropertyToSave)
        reducedTable.to_csv(filenameToSave, index=False)

    def columnsWithLowerCorrelation(self, valuesToCorrelate, corAgainstValues, rThreshold):
        nrOfCol = valuesToCorrelate.shape[1]
        correlations = np.zeros(nrOfCol)
        for i in range(nrOfCol):
            if len(corAgainstValues.shape) == 2:
                r = self.pooledRs(corAgainstValues, valuesToCorrelate.iloc[:, i], [0, 1,  2, 3, 4, 5]) #, [1,0], [5,4] could also test against difference in area or perimeter
            else:
                r, p = scipy.stats.pearsonr(corAgainstValues, valuesToCorrelate.iloc[:, i])
            correlations [i] = r
        columnsToKeep = np.where(correlations < rThreshold)[0]
        return columnsToKeep

    def pooledRs(self, corAgainst, valuesToCorrelate, corAgainstIndices):
        allRs = []
        for i in corAgainstIndices:
            if type(i) == list:
                r, p = scipy.stats.pearsonr(corAgainst.iloc[:, i[0]]-corAgainst.iloc[:, i[1]], valuesToCorrelate)
            else:
                r, p = scipy.stats.pearsonr(corAgainst.iloc[:, i], valuesToCorrelate)
            allRs.append(r)
        return np.max(allRs)

    def saveBasedOnOhterCorrelationFolder(self, folderToSave, featurePropertyName,
                             featurePropertyToLoad, keepFromFolder):
        filenameKeepFrom = Path(keepFromFolder).joinpath(featurePropertyName, f"combinedFeatures_{featurePropertyName}_notnormalised.csv")
        keepFromTable = pd.read_csv(filenameKeepFrom)
        columnNamesToKeep = list(keepFromTable.columns)
        filenameToLoad = folderToSave + "{}/combinedFeatures_{}_notnormalised.csv".format(featurePropertyToLoad, featurePropertyToLoad)
        table = pd.read_csv(filenameToLoad)
        reducedTable = table.loc[:, columnNamesToKeep]
        filenameToSave = folderToSave + "{}/combinedFeatures_{}_notnormalised.csv".format(featurePropertyName, featurePropertyName)
        reducedTable.to_csv(filenameToSave, index=False)

        # used by other code to create topo features
    def zNormalise(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data-mean)/std

    def createNewCombinedFeatureTable(self, baseFolder, concatTablename, calcDiff=False, calcAbsDiff=False,
                                      calcRatio=False, addDefNeighbourFeatures=False,
                                      normalisePerTissue=False, printTableName=True):
        newTableName = "{}Par{}{}.csv"
        table = pd.read_csv(baseFolder+concatTablename)
        cellIdentifiers = table.iloc[:, :4].copy(deep=True)
        features = table.iloc[:, 4:].copy(deep=True)
        nrOfRows = features.shape[1]
        rowToSplit = nrOfRows//2
        #  :rowToSplit is neighbour cell features; rowToSplit: is dividing cell features
        if calcDiff:
            featureId = "diff"
            newFeatures = features.iloc[:, :rowToSplit].to_numpy() - features.iloc[:, rowToSplit:].to_numpy()
        if calcAbsDiff:
            featureId = "absDiff"
            newFeatures = np.abs(features.iloc[:, :rowToSplit].to_numpy() - features.iloc[:, rowToSplit:].to_numpy())
        elif calcRatio:
            featureId = "ratio"
            newFeatures = features.iloc[:, :rowToSplit].to_numpy() / features.iloc[:, rowToSplit:].to_numpy()
        else:
            featureId = "default"
            newFeatures = features.iloc[:, :rowToSplit].to_numpy()
        newColNames = [colName.replace("_n", "") + "_" + featureId for colName in features.iloc[:, :rowToSplit].columns]
        newFeatures = pd.DataFrame(newFeatures, columns=newColNames)

        if addDefNeighbourFeatures:
            concatFeatureName = "_concat"
            newFeatures = pd.concat([newFeatures, features.iloc[:, :rowToSplit]], axis=1)
        else:
            concatFeatureName = ""
        # kick out nan's
        if normalisePerTissue:
            normalisePerTissueName = "_normPerTissue"
            plantNames = cellIdentifiers.iloc[:, 0]
            timePoints = cellIdentifiers.iloc[:, 1]
            for uniquePlant in np.unique(plantNames):
                for timePoint in np.unique(timePoints):
                    selectedRows = np.isin(plantNames, uniquePlant) & np.isin(timePoints, timePoint)
                    selectedRows = np.where(selectedRows)[0]
                    data = newFeatures.iloc[selectedRows, :].copy(deep=True)
                    data = zNormalise(data)
                    newFeatures.iloc[selectedRows, :] =  data
        else:
            normalisePerTissueName = "_notnormalised"
        newTable = pd.concat([cellIdentifiers, newFeatures], axis=1)
        newTableName = newTableName.format(featureId, concatFeatureName, normalisePerTissueName)
        if printTableName:
            print(newTableName)
        newTableName = baseFolder + newTableName
        newTable.to_csv(newTableName, index=False)

    def doTopoData(self):
        concatTablename = "defaultPar_concat_notnormalised.csv"
        baseFolder = "Data/WT/topoPredData/manuallyCreated/"
        addDefNeighbourFeatures = True
        calcAbsDiff = False
        calcRatio = False
        normalisePerTissue = False
        self.createNewCombinedFeatureTable(baseFolder, concatTablename,
                                          calcDiff=True,
                                          addDefNeighbourFeatures=addDefNeighbourFeatures,
                                          normalisePerTissue=normalisePerTissue)

def main():
    divDataArgs = {"dataFolder":"Data/WT/",
                   "folderToSave":"Data/WT/divEventData/",
                   "plantNames":["P1", "P2", "P5", "P6", "P8"],
                   "estimateFeatures":True,
                   "estimateLabels":True,
                   "useManualCentres":True}
    topoDataArgs = {"dataFolder":"Data/WT/",
                   "folderToSave":"Data/WT/topoPredData/diff/",
                   "plantNames":["P1", "P2", "P5", "P6", "P8"],
                   "estimateFeatures":True,
                   "estimateLabels":True,
                   "useManualCentres":True,
                   "useTopoCreator":True}
    CreateFeatureSets(**divDataArgs)

def ktnMain():
    createDivEventData = False
    centralCellsDict = {"ktnP1": [ [], [3839, 3959] ],
                        "ktnP2": [ [23], [424, 426, 50] ],
                        "ktnP3": [ [29, 199,527], [424, 28, 431] ] }
    ktnDivDataArgs = {"dataFolder": "Data/ktn/",
                   "folderToSave": "Data/ktn/divEventData/",
                   "plantNames": ["ktnP1", "ktnP2", "ktnP3"],
                   "estimateFeatures": True,
                   "estimateLabels": True,
                   "useManualCentres": True,
                   "timePointsPerPlant": 3,
                   "takeCorrelationFromDifferentFolder":"Data/WT/divEventData/manualCentres/",
                   "keepFromFolder":"Data/WT/divEventData/manualCentres/"}
    tktnTopoDataArgs = {"dataFolder":"Data/ktn/",
                   "folderToSave":"Data/ktn/topoPredData/diff/",
                   "plantNames":["ktnP1", "ktnP2", "ktnP3"],
                   "estimateFeatures": True,
                   "estimateLabels": True,
                   "useManualCentres": True,
                   "timePointsPerPlant": 3,
                   "useTopoCreator": True,
                   "takeCorrelationFromDifferentFolder":"Data/WT/topoPredData/diff/manualCentres/",
                   "keepFromFolder":"Data/WT/topoPredData/diff/manualCentres/"}
    if createDivEventData:
        CreateFeatureSets(skipEmptyCentrals=True, centralCellsDict=centralCellsDict,
                          **ktnDivDataArgs)
    else:
        CreateFeatureSets(skipEmptyCentrals=True, centralCellsDict=centralCellsDict,
                          **tktnTopoDataArgs)

if __name__ == '__main__':
    ktnMain()
    # main()
