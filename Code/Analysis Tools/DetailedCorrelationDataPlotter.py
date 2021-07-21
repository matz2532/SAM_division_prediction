import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats
import pandas as pd
import seaborn

sys.path.insert(0, "./Code/Classifiers/")
from DuplicateDoubleRemover import DuplicateDoubleRemover

class DetailedCorrelationDataPlotter (object):

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
        else:
            r = np.delete(self.givenCorrelation, duplicateColIdx)
            argmax = np.argmax(np.abs(r))
            self.plotCorrelations(r, savefig=True, filenameToSave=self.filenameToSaveCorrelationPlot)

    def deleteColdIdx(self, colToRemove):
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
        if not rThreshold is None:
            print("{} / {} Pearson correlation coefficients being higher than {}.".format(np.sum(r > rThreshold), len(r), rThreshold))
        fig, ax = plt.subplots()
        self.doScatterPlot(self.dataPoints1[:, argmax], self.dataPoints2[:, argmax], ax,
                           color="g", labels=self.colNames[argmax], savefig=True)
        self.plotCorrelations(r, savefig=True, filenameToSave=self.filenameToSaveCorrelationPlot)
        loadIdxToRemove = np.where(np.isin(self.colNames, "load centrality"))[0]
        print(self.colNames, loadIdxToRemove)
        self.doScatterPlotsExcept([argmax, loadIdxToRemove], r)

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
                      savefig=False, filenameToSave="highestCor.png", boxLoc=""):
        # seaborn.regplot(x, y)
        ax.scatter(x, y, marker=marker, c=color, alpha=0.8)
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
            boxX = maxX-(maxX-minX)/3
            boxY = minY+(maxY-minY)/2.3
            bbox = {'facecolor': 'grey', 'alpha': 0.5, 'pad': 10, "lw":1}
        m_rounded = self.appropriatelyRound(m)
        b_rounded = self.appropriatelyRound(b)
        r_rounded = self.appropriatelyRound(r)
        ax.text(boxX, boxY, 'f(x) = {0:#.2g}*x+{1:#.2g}\nr = {2:#.2g}'.format(m, b, r), style='normal', fontsize=12,
                bbox=bbox, multialignment="center")
        ax.plot(np.unique(minMaxX), np.poly1d((m, b))(np.unique(minMaxX)), c="black", lw=1)
        ax.set_title(labels, {'fontsize':"x-large"})
        ax.set_xlabel("observed value")
        ax.set_ylabel("predicted value")
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
        txt = ""
        for i in self.colNames[argsort]:
            i = i.replace("centrality", "c")
            i = i.replace("on 2 neighborhood", "2nd n")
            txt += i+"\n"
        print(txt)
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
                    boxLoc="top-left")
        plt.savefig(self.baseFolder + "Figure 4 Sups part{}.png".format(j), bbox_inches="tight")
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

    def comparativeBarPlot(self, firstValues, secondValues, addAverage=False,
                           filenameToSave=None, baseFolder="",
                           colNames=None, printOutColName=True, showPlot=True,
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
        colorColumnNumbers = len(firstColor)
        stackedColor = np.stack([np.full((nrOfValues, colorColumnNumbers), firstColor), np.full((nrOfValues, colorColumnNumbers), secondColor)], axis=1)
        stackedValues = np.stack([firstValues, secondValues], axis=1)
        colArgMax = np.argmax(stackedValues, axis=1)
        firstValueSet =[stackedValues[i, argmax] for i, argmax in enumerate(colArgMax)]
        secondValueSet =[stackedValues[i, np.invert(argmax)] for i, argmax in enumerate(colArgMax)]
        firstColorSet =[stackedColor[i, argmax] for i, argmax in enumerate(colArgMax)]
        secondColorSet =[stackedColor[i, np.invert(argmax)] for i, argmax in enumerate(colArgMax)]

        fig, ax = plt.subplots()
        y_pos = np.arange(nrOfValues)
        # ax.barh(y_pos, firstValueSet, align='center', color=firstColorSet)
        # ax.barh(y_pos, secondValueSet, align='center', color=secondColorSet)
        ax.barh(y_pos, firstValues, align='center', color=firstColor)
        ax.barh(y_pos, secondValues, align='center', color=secondColor, alpha=0.7)
        ax.set_xlabel("Pearson correlation coefficent")
        ax.set_yticks(y_pos)
        if  not colNames is None:
            ax.set_yticklabels(colNames)
            if printOutColName:
                txt = ""
                for i in colNames:
                    i = i.replace("centrality", "c")
                    i = i.replace("on 2 neighborhood", "2nd n")
                    txt += i+"\n"
                print(txt)
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

def main():
    sys.path.insert(0, "./Code/Propagation/")
    from RandomisedTissuePrediction import printMeanCorrelation
    colNames = pd.read_csv("./Data/WT/divEventData/manualCentres/topology/combinedFeatures_topology_notnormalised.csv").columns.to_numpy()[3:]
    # print(colNames)
    baseFolder = "Results/DivAndTopoApplication/"
    actualFeatures = np.load("./{}actualFeatures.npy".format(baseFolder))
    predFeatures = np.load("./{}predFeatures.npy".format(baseFolder))
    random = [None, "fullyRandom", "improvedRandom"][0]
    nrOfRepetitions = 100
    folderToLoadRandomCor = "Results/DivAndTopoApplication/Random/"
    if random is None:
        filenameToSaveCorrelationPlot = "corBarPlot.png"
        givenCorrelation = None
    elif random == "fullyRandom":
        filenameToSaveCorrelationPlot = "corBarPlot_fullyRandom.png"
        folderToLoadRandomCor += "FullyRandom/"
        givenCorrelation = printMeanCorrelation(nrOfRepetitions,
                                                folderToLoad=folderToLoadRandomCor,
                                                printOut=False)
    elif random == "improvedRandom":
        filenameToSaveCorrelationPlot = "corBarPlot_improvedRandom.png"
        folderToLoadRandomCor += "Realistic/"
        givenCorrelation = printMeanCorrelation(nrOfRepetitions,
                                                folderToLoad=folderToLoadRandomCor,
                                                printOut=False)
        givenCorrelation = np.asarray(givenCorrelation)
    myDetailedCorrelationDataPlotter = DetailedCorrelationDataPlotter(actualFeatures, predFeatures, colNames, baseFolder,
                                givenCorrelation=givenCorrelation,
                                filenameToSaveCorrelationPlot=filenameToSaveCorrelationPlot)
    myDetailedCorrelationDataPlotter.analyseDataPoints()

def testMain():
    baseFolder = "Results/DivAndTopoApplication/"
    filenameToSave = "propagationCorrelationsComparingWith{}.png"
    useFirst = False
    r = [0.8012599137646944, 0.7094876298482123, 0.6739029353003783, 0.6062687494385892, 0.6061962964973668, 0.5202427837097923, 0.5139619908103954, 0.5078058753445247, 0.49672226863991037, 0.4574032098665024, 0.4574032098665024, 0.4166512363995707, 0.4003791802201093, 0.3413708248255878, 0.3352235129325244, 0.30683542726811036, 0.29799806756472424, 0.2948589013287368]
    if useFirst:
        # divide all predicted non-dividing cells
        tag = "_DividingAllPredictedNonDividing"
        rComparison = [0.5226668883489629, 0.44040474180175027, 0.4354098621321026, 0.6256362308179784, 0.43156492312457945, 0.7548597607792287, 0.1967149244173097, 0.7489812672343444, 0.6706057440056719, 0.5010677381978935, 0.5010677381978935, 0.4377185485001872, 0.3442277400469092, 0.2200803353905863, 0.5975689994842004, 0.5359966939276773, 0.014364861688750577, -0.09214908403422475]
    else:
        # # randomly divide percentage of predicted non-dividing cellscells
        tag = "_RandomlyDividePercentageOfPredictedNonDividing"
        rComparison = [0.7809168081766118, 0.6807925034113907, 0.6898028652278158, 0.7131269716740737, 0.7076859270721316, 0.6862206031088972, 0.31352656498027887, 0.6827619679402527, 0.6558252219799935, 0.5622226775047566, 0.5622226775047566, 0.5483043013022503, 0.4050697796763687, 0.2899819452503018, 0.4137609679699336, 0.3208272155271749, 0.1499374505206079, 0.06242164482421148]
    argSort = np.argsort(r)[::-1]
    filenameToSave = baseFolder + filenameToSave.format(tag)
    DetailedCorrelationDataPlotter().comparativeBarPlot(np.asarray(r)[argSort], np.asarray(rComparison)[argSort],
                                                        filenameToSave=filenameToSave)

if __name__ == '__main__':
    # testMain()
    main()
