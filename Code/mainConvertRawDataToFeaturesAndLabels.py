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
    tasks = [[True, False], [True, True], [False, False], [False, True]]
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

def mainConvertFloralMeristemRawDataToFeaturesAndLabels():
    WtCentralCellsDict = {"p4 FM1" : [[21523], [21962, 21944], [22124, 22343, 22329], [22970, 22645, 22961], [24041, 24232]],
                          "p4 FM2" : [[164], [29886, 29856], [30336], [31576, 31129, 30829], [32691, 32822, 32823 ]]}
    wtDivDataArgs = {"dataFolder":"Data/floral meristems/WT/",
                   "folderToSave":"Data/floral meristems/WT/divEventData/",
                   "plantNames":["p4 FM1", "p4 FM2"],
                   "centerRadius":20,
                   "estimateFeatures":True,
                   "estimateLabels":True,
                   "useManualCentres":True,
                   "keepFromFolder":"Data/WT/divEventData/manualCentres/"}
    wtTopoDataArgs = {"dataFolder":"Data/floral meristems/WT/",
                   "folderToSave":"Data/floral meristems/WT/topoPredData/diff/",
                   "plantNames":["p4 FM1", "p4 FM2"],
                   "centerRadius":20,
                   "estimateFeatures":True,
                   "estimateLabels":True,
                   "useManualCentres":True,
                   "useTopoCreator":True,
                   "keepFromFolder":"Data/WT/topoPredData/diff/manualCentres/"}
    ktnCentralCellsDict = {"p3 FM6 flower 1" : [[13707, 13331], [12893, 12892], [14326, 14335, 14079], [], []],
                           "p3 FM6 flower 2" : [[43790, 44025], [43596, 43597], [19672, 19286], [], []],
                           "p3 FM6 flower 3" : [[], [20610], [20654, 20652, 20716], [21902, 22115, 22567], [40529, 39982]]}
    ktnDivDataArgs = {"dataFolder": "Data/floral meristems/ktn/",
                   "folderToSave": "Data/floral meristems/ktn/divEventData/",
                   "plantNames": ["p3 FM6 flower 1", "p3 FM6 flower 2", "p3 FM6 flower 3"],
                   "centerRadius":20,
                   "estimateFeatures": True,
                   "estimateLabels": True,
                   "useManualCentres": True,
                   "timePointsPerPlant": 5,
                   "keepFromFolder":"Data/WT/divEventData/manualCentres/"}
    tktnTopoDataArgs = {"dataFolder":"Data/floral meristems/ktn/",
                   "folderToSave":"Data/floral meristems/ktn/topoPredData/diff/",
                   "plantNames":["p3 FM6 flower 1", "p3 FM6 flower 2", "p3 FM6 flower 3"],
                   "centerRadius":20,
                   "estimateFeatures": True,
                   "estimateLabels": True,
                   "useManualCentres": True,
                   "timePointsPerPlant": 5,
                   "useTopoCreator": True,
                   "keepFromFolder":"Data/WT/topoPredData/diff/manualCentres/"}
    tasks = [[True, False], [True, True], [False, False], [False, True]]
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
        CreateFeatureSets(skipEmptyCentrals=True, centralCellsDict=centralCellsDict, setRange=[7],
                          **dataArgs)

if __name__== "__main__":
    # mainConvertRawDataToFeaturesAndLabels()
    mainConvertFloralMeristemRawDataToFeaturesAndLabels()
