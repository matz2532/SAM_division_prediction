import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sys

from matplotlib import gridspec
from statsmodels.stats.multitest import multipletests

class CorrelationHeatMapDisplayer (object):

    # plot correlation of features with area (Sup. Fig. 2)
    def __init__(self, features, featureColumns, featureColIdxToCorrelateAgainst=0):
        self.selectedFeatures = features.iloc[:, featureColumns]
        self.columns = list(self.selectedFeatures.columns)
        self.corFeature = self.selectedFeatures.iloc[:, featureColIdxToCorrelateAgainst].to_numpy()
        self.selectedFeatures = self.selectedFeatures.drop(self.columns[featureColIdxToCorrelateAgainst], axis=1)
        self.columns = list(self.selectedFeatures.columns)
        self.selectedFeatures = self.selectedFeatures.to_numpy()
        self.correlations, self.pValues = self.calcCorrelation()
        self.featureCorTable = pd.DataFrame({"Feature name":self.columns, "Pearson correlation coefficient":self.correlations, "p-value":self.pValues})
        self.plotHeatMap(self.correlations)

    def calcCorrelation(self):
        nrOfCol = self.selectedFeatures.shape[1]
        correlations = np.zeros(nrOfCol)
        pValues = np.zeros(nrOfCol)
        for i in range(nrOfCol):
            r, p = scipy.stats.pearsonr(self.corFeature, self.selectedFeatures[:, i])
            correlations [i] = r
            pValues[i] = p
        pValues = list(multipletests(pValues, method='fdr_bh')[1])
        return correlations, pValues

    def plotHeatMap(self, values, splits=4, printHighestCor=False, simpleHeatmap=False):
        nrOfFeatures = len(values) // splits
        correlationMatrix = np.zeros((nrOfFeatures, splits))
        idx = 0
        for i in range(splits):
            for j in range(nrOfFeatures):
                correlationMatrix[j, i] = values[idx]
                idx += 1
        correlationMatrix = correlationMatrix[1:, :]
        if printHighestCor:
            print(self.columns[np.argmax(np.abs(values))], np.max(values))
        identicalRows = self.identifyIdenticalRows(correlationMatrix)
        if simpleHeatmap or len(identicalRows) == 0:
            sns.heatmap(correlationMatrix, cmap="coolwarm", vmin=-1, vmax=1, #winter_r #"Reds"
                        xticklabels=False, yticklabels=False, annot=True,
                        linewidths=0.01, linecolor="black")
        else:
            self.plotHeatMapWithMergedIdenticalRows(correlationMatrix, identicalRows)

    def identifyIdenticalRows(self, twoDArray):
        identicalRows = []
        for i in range(twoDArray.shape[0]):
            absDiff = np.abs(twoDArray[i, :]-twoDArray[i, 0])
            allRowEntriesTheSame = np.all(absDiff < 1e-14)
            if allRowEntriesTheSame:
                identicalRows.append(i)
        return identicalRows

    def plotHeatMapWithMergedIdenticalRows(self, correlationMatrix, identicalRows):
        nrOfRows = correlationMatrix.shape[0]
        continiousSegments, isSegmentIdentical = self.determineContinousPlots(nrOfRows, identicalRows)
        nrows = len(continiousSegments)
        height_ratios = [len(i) for i in continiousSegments]
        gs = gridspec.GridSpec(nrows, 1, height_ratios=height_ratios)
        plt.subplots_adjust(hspace=.0)
        for i, segment in enumerate(continiousSegments):
            ax = plt.subplot(gs[i])
            values = correlationMatrix[segment, :]
            if isSegmentIdentical[i]:
                values = values[:, 0]
                if len(values.shape) < 2:
                    values = values.reshape(len(values), 1)
            sns.heatmap(values, ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, annot=True, cbar=False,
                linewidths=0.01, linecolor="black")

    def determineContinousPlots(self, nrOfRows, identicalRows):
        allContiniousSegments = []
        isSegmentIdentical = []
        lastContiniousSegment = [0]
        isIdentical = lastContiniousSegment[0] in identicalRows
        isLastIdentical = isIdentical
        for i in range(1, nrOfRows):
            isIdentical = i in identicalRows
            if isIdentical == isLastIdentical:
                lastContiniousSegment.append(i)
            else:
                allContiniousSegments.append(lastContiniousSegment)
                isSegmentIdentical.append(isLastIdentical)
                lastContiniousSegment = [i]
            isLastIdentical = isIdentical
        allContiniousSegments.append(lastContiniousSegment)
        isSegmentIdentical.append(isLastIdentical)
        return allContiniousSegments, isSegmentIdentical

    def SaveFeatureCorrelationTable(self, filenameToSave, featureNamesToIgnore=None,
                                    replacementDict={"replace":[".1", ".2", ".3"],
                                                     "with":[" weighted by area",
                                                             " weighted by shared wall",
                                                             " weighted by distance"]}
                                    ):
        featureColTable = self.featureCorTable
        if not featureNamesToIgnore is None:
            featureNames = self.featureCorTable.iloc[:, 0]
            isFeatureNameSelected = np.isin(featureNames, featureNamesToIgnore, invert=True)
            featureIdxToSelect = np.where(isFeatureNameSelected)[0]
        else:
            featureIdxToSelect = np.arange(featureColTable.shape[0])
        if not replacementDict is None:
            replace = replacementDict["replace"]
            replacement = replacementDict["with"]
            featureNames = featureColTable.iloc[:, 0]
            newFeatureNames = []
            for feature in featureNames:
                for i in range(len(replace)):
                    feature = feature.replace(replace[i], replacement[i])
                newFeatureNames.append(feature)
            featureColTable.iloc[:, 0] = newFeatureNames
        argSortCor = np.argsort(featureColTable.iloc[featureIdxToSelect, 1])
        featureIdxToSelect = featureIdxToSelect[argSortCor[::-1]]
        featureColTable.iloc[featureIdxToSelect, :].to_csv(filenameToSave, index=False)

def analyseCorrelationMAtrix(correlationMatrixFilename="Results/divEventData/correlationMatrixWithArea.npy"):
    correlationMatrix = np.load(correlationMatrixFilename)
    rowsToExclude = [0, 9, 12, 13, 14, 15]
    singleCors = correlationMatrix[rowsToExclude[1:], 0]
    correlationMatrix = np.delete(correlationMatrix, rowsToExclude, axis=0)
    correlationValues = np.concatenate([correlationMatrix, singleCors], axis=None)
    absCorrelationValues = np.abs(correlationValues)
    print(np.mean(absCorrelationValues<0.5))
    print(np.mean(absCorrelationValues<0.7))
    # sns.distplot(correlationValues, hist=True, kde=True,
    #          bins=10, color = 'darkblue',
    #          hist_kws={'edgecolor':'black'},
    #          kde_kws={'linewidth': 4})
    # plt.show()

def mainSaveCorrelation(baseFeatureFolder="Data/WT/divEventData/manualCentres/", savePlotFolder="",
                        featureStartIdx=3, featureColIdxToCorrelateAgainst=0):
    sys.path.insert(0, "./Code/Classifiers/")
    from DuplicateDoubleRemover import DuplicateDoubleRemover
    from pathlib import Path
    featureFileName = baseFeatureFolder + "topoAndBio/combinedFeatures_topoAndBio_notnormalised.csv"
    filenameToSave = savePlotFolder + "topoFeatureCorrelationWithArea.png"
    features = pd.read_csv(featureFileName)
    featureColumns = np.arange(featureStartIdx, len(features.columns))
    myDuplicateDoubleRemover = DuplicateDoubleRemover(features.iloc[:, featureColumns])
    duplicateColIdx = myDuplicateDoubleRemover.GetDuplicateColIdx()
    duplicateFeatureNames = np.asarray(list(features.iloc[:, featureColumns].columns))[duplicateColIdx]
    myHeatMapper = CorrelationHeatMapDisplayer(features, featureColumns, featureColIdxToCorrelateAgainst)
    pathOfCorTableResults = Path(filenameToSave)
    myHeatMapper.SaveFeatureCorrelationTable(pathOfCorTableResults.with_suffix(".csv"),
                                             featureNamesToIgnore=duplicateFeatureNames)
    plt.savefig(filenameToSave, bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == '__main__':
    mainSaveCorrelation()
