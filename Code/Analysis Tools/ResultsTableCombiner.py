import numpy as np
import pandas as pd

class ResultsTableCombiner (object):

    baseResultsTable=None
    extendedResultsTable=None
    verbosity=0

    def __init__(self, baseResultsFolder=None, classifierFolderExtension="", resultsTableName="resultsWithTesting.csv"):
        self.resultsTableName = resultsTableName
        self.baseResultsFolder = baseResultsFolder
        self.classifierFolderExtension = classifierFolderExtension

    def CreateBaseResultsTable(self, featureSetName, baseResultsTable=None, baseResultsTableFolder=None,
                               renameTrainMeanIdxTo="train mean", renameTestMeanIdxTo="test WT SAM"):
        if baseResultsTableFolder is None and not self.baseResultsFolder is None:
            baseResultsTableFolder = self.baseResultsFolder + featureSetName + "/" + self.classifierFolderExtension
        assert not baseResultsTable is None or not baseResultsTableFolder is None, "Either give a baseResultsTable or the baseResultsTableFolder to create the base result table."
        # load table
        if baseResultsTable is None:
            baseResultsTable = pd.read_csv(baseResultsTableFolder + self.resultsTableName, index_col=0)
        # rename test mean
        baseResultsTable = self.renameIdxOfTableFromTo(baseResultsTable, fromIdx="mean", toIdx=renameTrainMeanIdxTo)
        baseResultsTable = self.renameIdxOfTableFromTo(baseResultsTable, fromIdx="test mean", toIdx=renameTestMeanIdxTo)
        # round values accordingly e.g., accuracy to 2 and auc 4 digits after comma
        roundedBaseTable = self.roundValues(baseResultsTable, exceptForColumnsDict={})
        # combine mean and std rows with +-
        self.baseResultsTable = self.mergeRowEntriesWithNextRow(roundedBaseTable, rowIdxToUse=renameTrainMeanIdxTo, symbolToCombineRows="±")
        self.baseResultsTable = self.mergeRowEntriesWithNextRow(self.baseResultsTable, rowIdxToUse=renameTestMeanIdxTo, symbolToCombineRows="±")

    def ExtendBaseResultsTable(self, featureSetName, additionalTestFolderWithKeys):
        self.extendedResultsTable = self.baseResultsTable.copy()
        for newTestRowName, resultsBaseFolder in additionalTestFolderWithKeys.items():
            # load table
            resultsBaseFolder += featureSetName + "/" + self.classifierFolderExtension
            extendingTable = pd.read_csv(resultsBaseFolder + self.resultsTableName, index_col=0)
            # round values accordingly e.g., accuracy to 2 and auc 4 digits after comma
            extendingTable = self.roundValues(extendingTable, exceptForColumnsDict={})
            # combine mean and std rows with +-
            extendingTable = self.mergeRowEntriesWithNextRow(extendingTable, rowIdxToUse="test mean", symbolToCombineRows="±")
            # include test values
            nrOfBaseResultTablecolumns = self.extendedResultsTable.shape[1]
            valuesToExtend = list(extendingTable.loc["test mean"])
            while len(valuesToExtend) < nrOfBaseResultTablecolumns:
                valuesToExtend.insert(0, "")
            self.extendedResultsTable.loc[newTestRowName] = valuesToExtend

    def SaveResultsTable(self, tableFilenameToSave="combined scenarios results.csv", useExtendedTable=None):
        if useExtendedTable is None:
            if self.extendedResultsTable is None:
                useExtendedTable = False
            else:
                useExtendedTable = True
        if useExtendedTable:
            assert not self.extendedResultsTable is None, "The extended results table is None, but you wanted it to save./nTry setting 'useExtendedTable' to False to save the base results table."
            self.extendedResultsTable.to_csv(tableFilenameToSave)
        else:
            assert not self.baseResultsTable is None, "The base results table is None, but you wanted it to save."
            self.baseResultsTable.to_csv(tableFilenameToSave)

    def renameIdxOfTableFromTo(self, baseResultsTable, fromIdx, toIdx, inplace=False):
        indices = np.asarray(baseResultsTable.index)
        isIdxFromIdx = indices == fromIdx
        if np.sum(isIdxFromIdx) == 1:
            indices[isIdxFromIdx] = toIdx
            baseResultsTable.index = indices
        elif np.sum(isIdxFromIdx) > 1:
            print(f"The index '{fromIdx}' was not changed to '{toIdx}' as the index was present more than once {indices}")
        else:
            print(f"The index '{fromIdx}' was not changed to '{toIdx}' as the index was not present in {indices}")
        return baseResultsTable

    def roundValues(self, table, baseRoundValue=2, exceptForColumnsDict={"train Auc":4, "val Auc":4, "Auc":4}):
        columns = np.asarray(table.columns)
        roundedTable = table.round(baseRoundValue).copy()
        for columnToExcept, valueToRoundException in exceptForColumnsDict.items():
            isSpecialColumn = np.isin(columns, columnToExcept)
            if np.any(isSpecialColumn):
                idxOfColumnToRound = np.where(isSpecialColumn)[0]
                roundedTable.iloc[:, idxOfColumnToRound] = table.iloc[:, idxOfColumnToRound].round(valueToRoundException)
            else:
                if self.verbosity >1:
                    print(f"Warning: The row {columnToExcept} was not present and could therfore not be rounded to {valueToRoundException}, select any of these: {columns}")
        return roundedTable

    def mergeRowEntriesWithNextRow(self, table, rowIdxToUse, symbolToCombineRows="±"):
        indices = np.asarray(table.index)
        isIdxFromIdx = indices == rowIdxToUse
        if np.any(isIdxFromIdx):
            rowIdxToUse = np.where(isIdxFromIdx)[0]
            selectedRowValues = table.iloc[rowIdxToUse, :].to_numpy().ravel()
            nextRowValues = table.iloc[rowIdxToUse + 1, :].to_numpy().ravel()
            combinedRowValues = [str(i) + symbolToCombineRows + str(j) for i, j in zip(selectedRowValues, nextRowValues)]
            table.iloc[rowIdxToUse, :] = combinedRowValues
            table.drop(index=indices[rowIdxToUse + 1], inplace=True)
        else:
            print(f"The row with the index '{rowIdxToUse}' does not exist only {indices} do exist.")
        return table

def mainCombineDivPredResults(saveUnderFolder="", baseResultsFolder="Results/divEventData/manualCentres/",
                              featureSets=["allTopos", "area", "topoAndBio", "topology", "lowCor0.3"],
                              additionalTestFolderWithKeys={
                                        "test ktn SAM" : "Results/ktnDivEventData/manualCentres/",
                                        "test WT floral" : "Results/floral meristems/WT/divEventData/manualCentres/",
                                        "test ktn floral" : "Results/floral meristems/ktn/divEventData/manualCentres/"
                                                            },
                              classifierFolderExtension="svm_k1h_combinedTable_l3f0n1c0bal0ex0/",
                              resultsTableName="resultsWithTesting.csv",
                              tableName="div pred combined results of {}.csv"):
    myCombiner = ResultsTableCombiner(baseResultsFolder=baseResultsFolder,
                                      classifierFolderExtension=classifierFolderExtension,
                                      resultsTableName=resultsTableName)
    for featureSetName in featureSets:
        finalCombinedTableFilename = saveUnderFolder + tableName.format(featureSetName)
        myCombiner.CreateBaseResultsTable(featureSetName)
        myCombiner.ExtendBaseResultsTable(featureSetName, additionalTestFolderWithKeys)
        myCombiner.SaveResultsTable(finalCombinedTableFilename)

def mainCombineTopoPredResults(saveUnderFolder="", baseResultsFolder="Results/topoPredData/diff/manualCentres/",
                               featureSets=["allTopos", "bio", "topoAndBio", "topology", "lowCor0.3"],
                               additionalTestFolderWithKeys={
                                        "test ktn SAM" : "Results/ktnTopoPredData/diff/manualCentres/",
                                        "test WT floral" : "Results/floral meristems/WT/topoPredData/diff/manualCentres/",
                                        "test ktn floral" : "Results/floral meristems/ktn/topoPredData/diff/manualCentres/"
                                                            },
                               classifierFolderExtension="svm_k1h_combinedTable_l3f0n1c0bal0ex1/",
                               resultsTableName="resultsWithTesting.csv",
                               tableName="topo pred combined results of {}.csv"):
    myCombiner = ResultsTableCombiner(baseResultsFolder=baseResultsFolder,
                                      classifierFolderExtension=classifierFolderExtension,
                                      resultsTableName=resultsTableName)
    for featureSetName in featureSets:
        finalCombinedTableFilename = saveUnderFolder + tableName.format(featureSetName)
        myCombiner.CreateBaseResultsTable(featureSetName)
        myCombiner.ExtendBaseResultsTable(featureSetName, additionalTestFolderWithKeys)
        myCombiner.SaveResultsTable(finalCombinedTableFilename)

if __name__ == '__main__':
    mainCombineDivPredResults()
    mainCombineTopoPredResults()
