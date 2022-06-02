import numpy as np

class ResultsTableCombiner (object):

    baseResultsTable=None
    extendedResultsTable=None

    def __init__(self):
        pass

    def CreateBaseResultsTable(self):
        # load table
        # rename test mean
        # round values accordingly e.g., accuracy to 2 and auc 4 digits after comma
        # combine mean and std rows with +-

    def ExtendBaseResultsTable(self):
        # load table
        # rename test mean
        # round values accordingly e.g., accuracy to 2 and auc 4 digits after comma
        # combine mean and std rows with +-
        # include test values at correct positions

    def SaveResultsTable(self, tableFilenameToSave="combined scenarios results.csv", useExtendedTable=None):
        if useExtendedTable is None:
            if extendedResultsTable is None:
                useExtendedTable = False
            else:
                useExtendedTable = True
        if useExtendedTable:
            assert not self.extendedResultsTable is None, "The extended results table is None, but you wanted it to save.\nTry setting 'useExtendedTable' to False to save the base results table."
            self.extendedResultsTable.to_csv(tableFilenameToSave)
        else:
            assert not self.baseResultsTable is None, "The base results table is None, but you wanted it to save."
            self.baseResultsTable.to_csv(tableFilenameToSave)

def mainCombineDivPredResults(saveUnderFolder="", baseResultsFolder="/Results/divEventData/manualCentres/",
                              featureSets=["allTopos", "area", "topoAndBio", "topology", "lowCor0.3"],
                              additionalTestFolderWithKeys={
                                        "test ktn SAM" : "Results/ktnDivEventData/manualCentres/",
                                        "test WT floral meristem" : "Results/floral meristems/WT/divEventData/manualCentres/",
                                        "test ktn floral meristem" : "Results/floral meristems/ktn/divEventData/manualCentres/"
                                                            },
                              classifierFolderExtension="svm_k1h_combinedTable_l3f0n1c0bal0ex0/",
                              tableName="div pred combined results of {}.csv"):
    for featureSetName in featureSets:
        tableResultsName = saveUnderFolder + tableName.format(featureSetName)
        ResultsTableCombiner()

def mainCombineTopoPredResults(saveUnderFolder="", baseResultsFolder="/Results/topoPredData/diff/manualCentres/",
                               featureSets=["allTopos", "area", "topoAndBio", "topology", "lowCor0.3"],
                               additionalTestFolderWithKeys={
                                        "test ktn SAM" : "Results/ktnTopoPredData/diff/manualCentres/",
                                        "test WT floral meristem" : "Results/floral meristems/WT/topoPredData/diff/manualCentres/",
                                        "test ktn floral meristem" : "Results/floral meristems/ktn/divEventData/diff/manualCentres/"
                                                            },
                               classifierFolderExtension="svm_k1h_combinedTable_l3f0n1c0bal0ex1/",
                               tableName="div pred combined results of {}.csv"):
    for featureSetName in featureSets:
        tableResultsName = saveUnderFolder + tableName.format(featureSetName)
        ResultsTableCombiner()

def main():
    myResultsTableCombiner = ResultsTableCombiner()

if __name__ == '__main__':
    main()
