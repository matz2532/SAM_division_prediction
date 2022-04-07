import sys

sys.path.insert(0, "./Code/Feature and Label Creation/")
from CreateFeatureSets import CreateFeatureSets

def mainConvertRawDataToFeaturesAndLabels():
    WtCentralCellsDict = {"P1":[[618, 467, 570], [5048, 5305], [5849, 5601], [6178, 6155, 6164], [6288, 6240]],
                        "P2":[[392], [553, 779, 527], [525], [1135], [1664, 1657]],
                        "P5":[[38], [585, 968, 982], [927, 1017], [1136], [1618, 1575, 1445]],
                        "P6":[[861], [], [], [2109, 2176], [2381]],
                        "P8":[[3241, 2869, 3044], [3421, 3657], [], [], [358, 189]],
                        "P9":[[1047, 721, 1048], [7303, 7533], [6735, 7129], [2160, 2228], [7366, 7236]],
                        "P10":[[1511, 1524], [7281, 7516, 7534], [], [7634, 7722, 7795, 7794], [1073, 1074, 892]],
                        "P11":[[1751], [9489], [9759, 9793], [3300, 3211, 3060], [3956, 3979]]}
    wtDivDataArgs = {"dataFolder":"Data/WT/",
                   "folderToSave":"Data/WT/divEventData/",
                   "plantNames":["P1", "P2", "P5", "P6", "P8", "P9", "P10", "P11"],
                   "estimateFeatures":True,
                   "estimateLabels":True,
                   "useManualCentres":True}
    wtTopoDataArgs = {"dataFolder":"Data/WT/",
                   "folderToSave":"Data/WT/topoPredData/diff/",
                   "plantNames":["P1", "P2", "P5", "P6", "P8", "P9", "P10", "P11"],
                   "estimateFeatures":True,
                   "estimateLabels":True,
                   "useManualCentres":True,
                   "useTopoCreator":True}
    ktnCentralCellsDict = {"ktnP1": [ [], [3839, 3959] ],
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
    tasks = [[True, False], [True, True], [False, False], [False, True]] # full task list
    tasks = [ [True, False], [False, False]]
    for usingWT, createDivData in tasks:
        if usingWT:
            centralCellsDict = WtCentralCellsDict
            if createDivData:
                dataArgs = wtDivDataArgs
            else:
                dataArgs = wtTopoDataArgs
        else:
            centralCellsDict = ktnCentralCellsDict
            if createDivData:
                dataArgs = ktnDivDataArgs
            else:
                dataArgs = tktnTopoDataArgs
        CreateFeatureSets(skipEmptyCentrals=True, centralCellsDict=centralCellsDict,
                          **dataArgs)

if __name__== "__main__":
    mainConvertRawDataToFeaturesAndLabels()
