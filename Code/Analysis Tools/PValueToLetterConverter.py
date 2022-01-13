import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as st
import sys

from networkx.algorithms.clique import find_cliques
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class PValueToLetterConverter (object):

    pValueTable=None

    def __init__(self, pValueTable=None):
        if not pValueTable is None:
            self.SetPValueTable(pValueTable)
            self.groupNames = np.unique(np.concatenate([self.pValueTable["group1"],self.pValueTable["group2"]]))
            self.groupNamesLetters = self.calcLettersForGroup(self.pValueTable)

    def GetGroupNameLetters(self):
        return self.groupNamesLetters

    def SetPValueTable(self, pValueTable):
        self.pValueTable = self.checkAndCorrectTableFormat(pValueTable)

    def checkAndCorrectTableFormat(self, pValueTable, alpha=0.05):
        assert isinstance(pValueTable, pd.DataFrame), "The pValueTable should be a pd.DataFrame, but is {}".format(type(pValueTable))
        columns = list(pValueTable.columns)
        assert "group1" in columns, "The column 'group1' is missing in {}".format(columns)
        assert "group2" in columns, "The column 'group2' is missing in {}".format(columns)
        isRejectColumnMissing = not "reject" in columns
        if isRejectColumnMissing:
            assert "p-value" in columns,  "You need to give the column 'reject' or 'p-value', not present in {}".format(columns)
            pValueTable["reject"] = pValueTable["p-value"] < alpha
        return pValueTable

    def calcLettersForGroup(self, pValueTable, useSpecificletters=None):
        lettersOfGroup = {}
        significanceNetwork = self.createSignificanceNetwork(pValueTable)
        nodesOfCliques = list(find_cliques(significanceNetwork))
        cliquesIds = self.calcUniqueCliquesLetter(nodesOfCliques)
        groupNamesLetters = {i : "" for i in self.groupNames}
        if not useSpecificletters is None:
             letters = useSpecificletters
        else:
            letters = np.concatenate([list(ascii_lowercase), list(ascii_uppercase)])
        for i, groupName in enumerate(self.groupNames):
            for j, clique in enumerate(nodesOfCliques):
                if groupName in clique:
                    round = i // len(letters)
                    if round == 0:
                        roundExtension = ""
                    else:
                        roundExtension = "{}".format(round)
                    groupNamesLetters[groupName] += letters[cliquesIds[j]] + roundExtension
        return groupNamesLetters

    def createSignificanceNetwork(self, pValueTable):
        significantGroupsDifferences = pValueTable[np.invert(pValueTable["reject"])]
        edges = list(zip(significantGroupsDifferences["group1"], significantGroupsDifferences["group2"]))
        network = nx.from_edgelist(edges)
        uniqueGroups = np.unique(pValueTable.iloc[:, :2])
        isGroupMissingAsNode = np.isin(uniqueGroups, list(network.nodes()), invert=True)
        if np.any(isGroupMissingAsNode):
            network.add_nodes_from(uniqueGroups[isGroupMissingAsNode])
        return network

    def calcUniqueCliquesLetter(self, nodesOfCliques):
        numberOfCliques = len(nodesOfCliques)
        cliquesIds = {}
        currentIdx = 0
        for groupName in self.groupNames:
            for i, clique in enumerate(nodesOfCliques):
                if groupName in clique:
                    if i not in cliquesIds:
                        cliquesIds[i] = currentIdx
                        currentIdx += 1
                if len(cliquesIds) == numberOfCliques:
                    break
        return cliquesIds

def main():
    myPValueToLetterConverter = PValueToLetterConverter()

if __name__ == '__main__':
    main()
