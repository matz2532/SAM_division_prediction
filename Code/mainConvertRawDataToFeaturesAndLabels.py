import sys

sys.path.insert(0, "./Code/Feature and Label Creation/")
from CreateFeatureSets import CreateFeatureSets

def mainConvertRawDataToFeaturesAndLabels():
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
                   "takeCorrelationFromDifferentFolder":"Data/WT/divEventData/manualCentres/"}
    tktnTopoDataArgs = {"dataFolder":"Data/ktn/",
                   "folderToSave":"Data/ktn/topoPredData/diff/",
                   "plantNames":["ktnP1", "ktnP2", "ktnP3"],
                   "estimateFeatures": True,
                   "estimateLabels": True,
                   "useManualCentres": True,
                   "timePointsPerPlant": 3,
                   "useTopoCreator": True,
                   "takeCorrelationFromDifferentFolder":"Data/WT/topoPredData/diff/manualCentres/"}
    if createDivEventData:
        CreateFeatureSets(skipEmptyCentrals=True, centralCellsDict=centralCellsDict,
                          **ktnDivDataArgs)
    else:
        CreateFeatureSets(skipEmptyCentrals=True, centralCellsDict=centralCellsDict,
                          **tktnTopoDataArgs)

if __name__== "__main__":
    mainConvertRawDataToFeaturesAndLabels()
