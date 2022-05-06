import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats
import pandas as pd
import seaborn

sys.path.insert(0, "./Code/Classifiers/")
from DuplicateDoubleRemover import DuplicateDoubleRemover

class DetailedCorrelationDataPlotter (object):

    correlationInformation=""

    # compare correlations of to data sets (Fig. 4D)
    def __init__(self, dataPoints1=None, dataPoints2=None, colNames=None, baseFolder="",
                 givenCorrelation=None, filenameToSaveCorrelationPlot="corBarPlot.png"):
        self.dataPoints1 = dataPoints1
        self.dataPoints2 = dataPoints2
        self.colNames = colNames
        self.baseFolder = baseFolder
        self.givenCorrelation = givenCorrelation
        self.filenameToSaveCorrelationPlot = filenameToSaveCorrelationPlot

    def analyseDataPoints(self):
        assert self.dataPoints1.shape == self.dataPoints2.shape, "The shapes of data points from 1 and 2 are different, {} != {}".format(dataPoints1.shape, dataPoints2.shape)
        duplicateColIdx = DuplicateDoubleRemover(pd.DataFrame(self.dataPoints1)).GetDuplicateColIdx()
        self.deleteColdIdx(duplicateColIdx)
        if self.givenCorrelation is None:
            self.plotDefaultDetailedCorrelations(duplicateColIdx)
            corInfoExtension = "vsRandom"
            if self.correlationInformation != "":
                filename = self.baseFolder + f"correlation{corInfoExtension}Information.txt"
                with open(filename, "w") as fh:
                    fh.write(self.correlationInformation)
        else:
            r = np.delete(self.givenCorrelation, duplicateColIdx)
            argmax = np.argmax(np.abs(r))
            self.plotCorrelations(r, savefig=True, filenameToSave=self.filenameToSaveCorrelationPlot)

    def deleteColdIdx(self, colToRemove):
        if len(colToRemove) > 0:
            self.dataPoints1 = np.delete(self.dataPoints1, colToRemove, axis=1)
            self.dataPoints2 = np.delete(self.dataPoints2, colToRemove, axis=1)
            self.colNames = np.delete(self.colNames, colToRemove)

    def plotDefaultDetailedCorrelations(self, duplicateColIdx, rThreshold=0.5):
        r, corPValues = self.correlateFeatures(self.dataPoints1, self.dataPoints2)
        pValues, CI = self.checkDifferences(self.dataPoints1, self.dataPoints2)
        corPValues = np.round(corPValues, 2)
        pValues = np.round(pValues, 2)
        for i in np.argsort(r)[::-1]:
            print(self.colNames[i], r[i], corPValues[i], pValues[i])
        argmax = np.argmax(np.abs(r))
        self.correlationInformation = ""
        if not rThreshold is None:
            countAboveThresholdText = "{} / {} Pearson correlation coefficients being higher than {}.".format(np.sum(r > rThreshold), len(r), rThreshold)
            self.correlationInformation += countAboveThresholdText + "\n"
            print(countAboveThresholdText)
        fig, ax = plt.subplots()
        self.doScatterPlot(self.dataPoints1[:, argmax], self.dataPoints2[:, argmax], ax,
                           color="g", labels=self.colNames[argmax], savefig=True, fontSize=18)
        self.doScatterPlotsExcept([argmax], r)

    def correlateFeatures(self, expectedFeatures, observedFeatures):
        correlations = []
        pValues = []
        for i in range(expectedFeatures.shape[1]):
            r, p = scipy.stats.pearsonr(expectedFeatures[:, i], observedFeatures[:, i])
            correlations.append(r)
            pValues.append(p)
        return np.asarray(correlations), np.asarray(pValues)

    def checkDifferences(self, dataPoints1, dataPoints2):
        pValues = []
        confidenceIntervalls = []
        for i in range(dataPoints1.shape[1]):
            t, p = scipy.stats.ttest_rel(dataPoints1[:, i], dataPoints2[:, i])
            pValues.append(p)
        return np.asarray(pValues), np.asarray(confidenceIntervalls)

    def doScatterPlot(self, x, y, ax, marker="o", color="g", showPlot=False, labels="",
                      savefig=False, filenameToSave="highestCor.png", boxLoc="", fontSize=10):
        matplotlib.rcParams.update({"font.size":fontSize})
        ax.scatter(x, y, marker=marker, c=color, alpha=0.3)
        m, b = np.polyfit(x, y, 1)
        minX, maxX = np.min(x), np.max(x)
        minY, maxY = np.min(y), np.max(y)
        minMaxRange = maxX - minX
        margin = minMaxRange * 0.07
        minXY = np.asarray(self.myround([minX, minY], base=minMaxRange/6, isFloor=True))-margin
        maxXY = np.asarray(self.myround([maxX, maxY], base=minMaxRange/6, isFloor=False))+margin
        minMaxX = np.asarray([minXY[0], maxXY[0]])
        r, p = scipy.stats.pearsonr(x, y)
        if boxLoc == "top-left":
            boxX = minX#-(maxX-minX)/2
            boxY = maxY-(maxY-minY)/5
            bbox = None
            if labels == "abs graph density 2nd n":
                boxY -= 0.01
            elif labels == "avg path length 2nd n":
                boxY += 0.035
        else:
            boxX = maxX-(maxX-minX)/2.3
            boxY = minY+(maxY-minY)/10
            bbox = {'facecolor': 'grey', 'alpha': 0.5, 'pad': 5, "lw":1}
        m_rounded = self.appropriatelyRound(m)
        b_rounded = self.appropriatelyRound(b)
        r_rounded = self.appropriatelyRound(r)
        ax.text(boxX, boxY, 'f(x) = {0:#.2g}*x+{1:#.2g}\nr = {2:#.2g}'.format(m, b, r), style='normal', fontsize=12,
                bbox=bbox, multialignment="center", size=fontSize)
        ax.plot(np.unique(minMaxX), np.poly1d((m, b))(np.unique(minMaxX)), c="black", lw=1)
        ax.set_title(labels, {'fontsize':fontSize})
        ax.set_xlabel("observed value", size=fontSize)
        ax.set_ylabel("predicted value", size=fontSize)
        ax.tick_params(axis="both", labelsize=fontSize)
        ax.set_xlim((minXY[0], maxXY[0]))
        ax.set_ylim((minXY[1], maxXY[1]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if savefig:
            plt.savefig(self.baseFolder + filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if showPlot:
                plt.show()

    def appropriatelyRound(self, value, minDecimalPlaces=2):
        if value >= 0.1:
            roundTo = minDecimalPlaces
        else:
            roundTo = np.abs(np.log10(np.min(value))-minDecimalPlaces)
        return np.round(value, int(roundTo))

    def plotCorrelations(self, r, showPlot=False, filenameToSave="corBarPlot.png", savefig=False):
        fig, ax = plt.subplots()
        argsort = np.argsort(r)[::-1]
        y_pos = np.arange(len(self.colNames))
        ax.barh(y_pos, r[argsort], align='center')
        ax.set_xlabel("Pearson correlation coefficent")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.colNames[argsort])
        ax.set_xlim((0, 1))
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        featureAbreviationNames = ""
        for i in self.colNames[argsort]:
            i = i.replace("centrality", "c")
            i = i.replace("on 2 neighborhood", "2nd n")
            featureAbreviationNames += i+"\n"
        self.correlationInformation += featureAbreviationNames
        if savefig:
            plt.savefig(self.baseFolder + filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if showPlot:
                plt.show()

    def myround(self, x, base=5, isFloor=True):
        if type(x) == list:
            x = np.asarray(x)
        if isFloor:
            return base * x/base
        else:
            return base * x/base

    def doScatterPlotsExcept(self, indexToExcept, correlationValues):
        self.deleteColdIdx(indexToExcept)
        colNames = [i.replace("centrality", "c") for i in self.colNames]
        colNames = [i.replace("on 2 neighborhood", "2nd n") for i in colNames]
        correlationValues = np.delete(correlationValues, indexToExcept)
        nrOfRows = self.dataPoints1.shape[0]
        nrOfCols = self.dataPoints1.shape[1]
        markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        colors = []
        sqrtOfCols = np.ceil(np.sqrt(nrOfCols)).astype(int)
        order = np.argsort(np.abs(correlationValues))[::-1]
        inOneGo = False
        if inOneGo:
            fig, ax = plt.subplots(nrows=sqrtOfCols, ncols=sqrtOfCols, figsize=(4*sqrtOfCols, 4*sqrtOfCols), dpi=50)
            plt.subplots_adjust(hspace=0.5)
            for i, idx in enumerate(order):
                x = i//sqrtOfCols
                y = i%sqrtOfCols
                ax_i = ax[x, y]
                self.doScatterPlot(self.dataPoints1[:, idx], self.dataPoints2[:, idx],
                        ax=ax_i, marker=markers[i], labels=colNames[idx],
                        boxLoc="top-left")
            plt.tight_layout()
            plt.show()
        else:
            j = 1
            for c in range(2):
                for r in range(2):
                    self.save4x4Subplots(c, r, sqrtOfCols, order, markers, colNames, j)
                    j += 1
        # plt.savefig(self.baseFolder + "Figure 4 Sups.png", bbox_inches="tight")
        # plt.close()

    def save4x4Subplots(self, c, r, sqrtOfCols, order, markers, colNames, j):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), dpi=100)
        plt.subplots_adjust(hspace=0.25, wspace=0.3)
        seclectedIdx = np.asarray([0,1,4,5])
        seclectedIdx +=  2*r + c*2*sqrtOfCols
        inidicesToPlot = order[seclectedIdx]
        for i, idx in enumerate(inidicesToPlot):
            x = i//2
            y = i%2
            ax_i = ax[x, y]
            self.doScatterPlot(self.dataPoints1[:, idx], self.dataPoints2[:, idx],
                    ax=ax_i, marker=markers[idx], labels=colNames[idx],
                    boxLoc="top-left", fontSize=14)
        plt.savefig(self.baseFolder + "Figure Sups part{}.png".format(j), bbox_inches="tight")
        plt.close()

    def doZNormalise(self, twoDArray, useParameters=None, returnParameter=False):
        if not useParameters is None:
            mean = useParameters[0]
            std = useParameters[1]
        else:
            mean = np.mean(twoDArray, axis=0)
            std = np.std(twoDArray, axis=0)
        twoDArray = (twoDArray-mean)/std
        if returnParameter:
            return [twoDArray, [mean, std]]
        else:
            return twoDArray

    def doMinMaxNorm(self, twoDArray):
        min = np.min(twoDArray, axis=0)
        max = np.max(twoDArray, axis=0)
        twoDArray = (twoDArray - min)/(max-min)
        return twoDArray

    def areColumnsTheSame(self, column1, column2):
        return np.all(column1 == column2)

    def comparativeBarPlot(self, firstValues, secondValues, secondValuesXerr, addAverage=False,
                           filenameToSave=None, baseFolder="",
                           colNames=None, trimColumnNames=True, printOutTrimmedColName=False,
                           showYLabels=True,
                           firstColor="blue", secondColor="grey", colorPaletteName="colorblind"):# deep,
        if addAverage:
            firstValues = np.concatenate([[np.mean(firstValues)], firstValues])
            secondValues = np.concatenate([[np.mean(secondValues)], secondValues])
        nrOfValues = len(firstValues)
        assert len(firstValues) == len(secondValues), "The values of the first and second vector need to be of the same length, {} != {}".format(len(firstValues), len(secondValues))
        assert (not colorPaletteName is None) or (not firstColor is None and not secondColor is None), "Either give a colot palette name or the respective colors."
        if not colorPaletteName is None:
            colorPallette = seaborn.color_palette(colorPaletteName)
            firstColor = colorPallette[0]
            secondColor = colorPallette[1]
        fig, ax = plt.subplots()
        y_pos = np.arange(nrOfValues)
        # colorColumnNumbers = len(firstColor)
        # stackedColor = np.stack([np.full((nrOfValues, colorColumnNumbers), firstColor), np.full((nrOfValues, colorColumnNumbers), secondColor)], axis=1)
        # stackedValues = np.stack([firstValues, secondValues], axis=1)
        # colArgMax = np.argmax(stackedValues, axis=1)
        # firstValueSet =[stackedValues[i, argmax] for i, argmax in enumerate(colArgMax)]
        # secondValueSet =[stackedValues[i, np.invert(argmax)] for i, argmax in enumerate(colArgMax)]
        # firstColorSet =[stackedColor[i, argmax] for i, argmax in enumerate(colArgMax)]
        # secondColorSet =[stackedColor[i, np.invert(argmax)] for i, argmax in enumerate(colArgMax)]
        # ax.barh(y_pos, firstValueSet, align='center', color=firstColorSet)
        # ax.barh(y_pos, secondValueSet, align='center', color=secondColorSet)
        ax.barh(y_pos, firstValues, align='center', color=firstColor)
        ax.barh(y_pos, secondValues, xerr=secondValuesXerr, align='center', color=secondColor, ecolor=secondColor, capsize=5, alpha=0.7)
        ax.set_xlabel("Pearson correlation coefficent")
        ax.set_yticks(y_pos)
        if not colNames is None:
            txt = ""
            trimmedColNames = []
            for i in colNames:
                i = i.replace("centrality", "c")
                i = i.replace("on 2 neighborhood", "2nd n")
                txt += i+"\n"
                trimmedColNames.append(i)
            if trimColumnNames:
                colNames = trimmedColNames
            if showYLabels:
                ax.set_yticklabels(colNames)
            if printOutTrimmedColName:
                print(txt)
            filename = baseFolder + f"correlationPredVsObsInformation.txt"
            with open(filename, "w") as fh:
                fh.write(txt)
        ax.set_xlim((0, 1))
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        if not filenameToSave is None:
            if baseFolder is None:
                self.baseFolder = baseFolder
            plt.savefig(baseFolder + filenameToSave, bbox_inches="tight")
            plt.close()
        else:
            if showPlot:
                plt.show()

def combinFeatures(baseFolder, plantNames):
    actualFeatures = []
    predFeatures = []
    for plant in plantNames:
        actualFeatures.append(np.load("./{}{}/actualFeatures.npy".format(baseFolder, plant)))
        predFeatures.append(np.load("./{}{}/predFeatures.npy".format(baseFolder, plant)))
    actualFeatures = np.concatenate(actualFeatures, axis=0)
    predFeatures = np.concatenate(predFeatures, axis=0)
    return actualFeatures, predFeatures

def plotFeaturesCorrelationOfPredVsObsPropagation():
    sys.path.insert(0, "./Code/Propagation/")
    # so actually dont load correlations to do correlation plots?
    from RandomisedTissuePrediction import calcMeanAndStdCorrelation
    plantNames = ["P2", "P9"]
    ignoreFeaturesIdx = [0]
    colNames = pd.read_csv("./Data/WT/divEventData/manualCentres/topology/combinedFeatures_topology_notnormalised.csv").columns.to_numpy()[3:]
    # print(colNames)
    baseFolder = "Results/DivAndTopoApplication/"
    random = [None, "fullyRandom", "improvedRandom"][0]
    nrOfRepetitions = 100
    folderToLoadRandomCor = "Results/DivAndTopoApplication/Random/"
    if random is None:
        filenameToSaveCorrelationPlot = "corBarPlot.png"
        givenCorrelation = None
    else:
        if random == "fullyRandom":
            filenameToSaveCorrelationPlot = "corBarPlot_fullyRandom.png"
            folderToLoadRandomCor += "FullyRandom/"
        elif  random == "improvedRandom":
            filenameToSaveCorrelationPlot = "corBarPlot_improvedRandom.png"
            folderToLoadRandomCor += "Realistic/"
        givenCorrelation, _ = calcMeanAndStdCorrelation(nrOfRepetitions,
                                               plantNames=plantNames,
                                               folderToLoad=folderToLoadRandomCor,
                                               printOut=False)
        givenCorrelation = np.asarray(givenCorrelation)
    actualFeatures, predFeatures = combinFeatures(baseFolder, plantNames)
    if ignoreFeaturesIdx:
        featuresToKeep = np.arange(len(colNames))
        featuresToKeep = featuresToKeep[np.isin(featuresToKeep, ignoreFeaturesIdx, invert=True)]
        colNames = colNames[featuresToKeep]
        actualFeatures = actualFeatures[:, featuresToKeep]
        predFeatures = predFeatures[:, featuresToKeep]
    myDetailedCorrelationDataPlotter = DetailedCorrelationDataPlotter(actualFeatures, predFeatures, colNames, baseFolder,
                                givenCorrelation=givenCorrelation,
                                filenameToSaveCorrelationPlot=filenameToSaveCorrelationPlot)
    myDetailedCorrelationDataPlotter.analyseDataPoints()

def compareModelPredictedVsRandomPropagationFeatureCorrelations(baseFolder="Results/DivAndTopoApplication/",
                filenameToSave="propagationCorrelationsComparingWith{}.png",
                featureFilename="Data/WT/divEventData/manualCentres/topology/combinedFeatures_topology_notnormalised.csv",
                saveUnderFolder=None, startingFeatureIdx=4, ignoreFeaturesIdx=[0]):
    if saveUnderFolder is None:
        saveUnderFolder = baseFolder
    r = np.load(baseFolder + "correlations.npy")
    if not featureFilename is None:
        featureNames = np.asarray(list(pd.read_csv(featureFilename).columns))[3:]
    else:
        featureNames = None
    # compare againts randomisation of divide all predicted non-dividing cells
    tag = "_DividingAllPredictedNonDividing"
    rComparison = np.load(baseFolder + "Random/FullyRandom/fullyReversedPrediction/meanCorrelations.npy")
    rComparisonXerr = np.load(baseFolder + "Random/FullyRandom/fullyReversedPrediction/stdCorrelations.npy")
    if ignoreFeaturesIdx:
        featuresToKeep = np.arange(len(r))
        featuresToKeep = featuresToKeep[np.isin(featuresToKeep, ignoreFeaturesIdx, invert=True)]
        r = r[featuresToKeep]
        rComparisonXerr = rComparisonXerr[featuresToKeep]
        rComparison = rComparison[featuresToKeep]
        if not featureNames is None:
            featureNames = featureNames[featuresToKeep]
    argSort = np.argsort(r)[::-1]
    if not featureNames is None:
        featureNames = featureNames[argSort]
    filenameToSave = filenameToSave.format(tag)
    DetailedCorrelationDataPlotter().comparativeBarPlot(np.asarray(r)[argSort], np.asarray(rComparison)[argSort],
                                                        secondValuesXerr=rComparisonXerr[argSort],
                                                        colNames=featureNames,
                                                        baseFolder=baseFolder,
                                                        filenameToSave=filenameToSave)

if __name__ == '__main__':
    compareModelPredictedVsRandomPropagationFeatureCorrelations()
    plotFeaturesCorrelationOfPredVsObsPropagation()
