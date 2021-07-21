import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

class FeatureDensityPlotting (object):

    def __init__(self, featureFilename, labelVectorFilename, runOnInit=False, selectedFeatures=None,
                nrOfSamplesPerTissue=None, baseFolder=None, seperateReplicates=False,
                timePointDeclaration=None, perReplicate=False, showWarnings=True,
                replicateId=None, index_col=None):
        self.features = self.loadTable(featureFilename, convertToNumpy=False, index_col=index_col)
        self.labels = self.loadTable(labelVectorFilename, index_col=index_col).flatten()
        self.featureNames = list(self.features.columns.values)
        self.selectedFeatures = self.setSelectedFeature(selectedFeatures)
        self.nrOfSamplesPerTissue = nrOfSamplesPerTissue
        self.baseFolder = baseFolder
        self.seperateReplicates = seperateReplicates
        self.nrOfTimepoints = 4
        self.timePointDeclaration = timePointDeclaration
        self.perReplicate = perReplicate
        self.replicateId = replicateId
        self.allFeaturesAppendix = "allFeaturesPooledOverAllSamples.png"
        self.perTimePointAppendix = "OverTimePoints_{}.png"
        self.perReplicateAppendix = "PerReplicate_{}.png"
        self.showWarnings = showWarnings
        if runOnInit:
            if nrOfSamplesPerTissue is None:
                if self.perReplicate:
                    self.plotPerReplicate()
                else:
                    self.plotAllFeaturesTogether()
            else:
                if self.replicateId is None:
                    self.plotFeaturesPerOverTimePoints()
                else:
                    self.plotFeaturesPerReplicate()

    def loadTable(self, filename, fillNa=True, convertToNumpy=True, index_col=None):
        if type(filename) is str:
            filename = pd.read_csv(filename, sep=",", index_col=index_col)
        if fillNa:
            filename.fillna(0, inplace=True)
        if convertToNumpy:
            filename = filename.to_numpy(copy=True)
        return filename

    def setSelectedFeature(self, selectedFeatures):
        if selectedFeatures is None:
            return self.featureNames
        elif type(selectedFeatures) == "str":
            return [selectedFeatures]
        return selectedFeatures

    def plotPerReplicate(self, nrOfTimepoints=4, nrOfReplicates=5, offset=0.4):
        assert nrOfTimepoints*nrOfReplicates == len(self.perReplicate), "The number of seperate replicates needs to be nrOfTimepoints times nrOfReplicates."
        nrOfSelecetedFeatures = len(self.perReplicate)
        nrOfRows = nrOfTimepoints
        nrOfColumns = nrOfReplicates
        plt.subplots_adjust(hspace=0.7, wspace=0.7)
        allCurrentPos, allNextPos = self.determineCurrentAndNextPosition(self.perReplicate)
        title = ""
        for feature in self.selectedFeatures:
            xmin = np.min(self.features[feature])-offset*np.max(np.abs(self.features[feature]))
            xmax = np.max(self.features[feature])+offset*np.max(np.abs(self.features[feature]))
            xlim = (10, 70)#(xmin, xmax)
            print(xlim)
            for i in range(nrOfSelecetedFeatures):
                plotPosition = 1 + (i % nrOfTimepoints) * nrOfReplicates + i // nrOfTimepoints#i+1
                currentPos = allCurrentPos[plotPosition-1]
                nextPos = allNextPos[plotPosition-1]
                idxOfTimePoint = np.arange(currentPos, nextPos)
                plt.subplot(nrOfRows, nrOfColumns, plotPosition)
                self.plotFeature(feature, idxOfTimePoint, title=title)
                plt.xlim(xlim)
                plt.xticks(np.arange(20, 70, 20))
            plt.legend(np.unique(self.labels))
            if self.baseFolder is None:
                plt.show()
            else:
                plt.savefig(self.baseFolder+self.allFeaturesAppendix, bbox_inches="tight")
                plt.close()

    def plotAllFeaturesTogether(self, filenameToSave=None):
        nrOfSelecetedFeatures = len(self.selectedFeatures)
        nrOfRows = np.ceil(np.sqrt(nrOfSelecetedFeatures))
        plt.subplots_adjust(hspace=0.7, wspace=0.7)
        for i in range(len(self.selectedFeatures)):
            plt.subplot(nrOfRows, nrOfRows, i+1)
            self.plotFeature(self.selectedFeatures[i])
        # plt.legend(np.unique(self.labels))
        if self.baseFolder is None and filenameToSave is None:
            plt.show()
        else:
            if not filenameToSave is None:
                plt.savefig(filenameToSave, bbox_inches="tight")
            else:
                plt.savefig(self.baseFolder + self.allFeaturesAppendix, bbox_inches="tight")
            plt.close()

    def plotFeature(self, currentFeature, selectedIdx=None, ax=None, showPlot=False,
                    title="{}", bw=None):
        featureIdx = np.where(self.featureNames == np.asarray(currentFeature))[0]
        uniqueLabels = np.unique(self.labels)
        if len(uniqueLabels) > 10 and showWarnings:
            print("WARNING in FeatureDensityPlotting: Are you sure you have more than ten unique label classes? Only supply the label classes per entry. Maybe your identifiers need to be set as indices.")
        for label in uniqueLabels:
            labelIdx = np.where(self.labels == label)[0]
            if not selectedIdx is None:
                labelIdx = labelIdx[np.isin(labelIdx, selectedIdx)]
            data = self.features.iloc[labelIdx, featureIdx].values.flatten()
            sns.kdeplot(data, shade=True)#, bw="scott")
        # plt.xlabel("density", fontsize = 15)
        # plt.ylabel(currentFeature, fontsize = 15)
        plt.title(title.format(currentFeature), fontsize = 8)
        if showPlot:
            plt.show()

    def plotFeaturesPerReplicate(self, offset=0.4):
        uniqueReplicates = np.unique(self.replicateId)
        nrOfPlots = len(uniqueReplicates) + 1
        for feature in self.featureNames:
            plt.subplots_adjust(hspace=0.7, wspace=0.7)
            xmin = np.min(self.features[feature])-offset*np.max(np.abs(self.features[feature]))
            xmax = np.max(self.features[feature])+offset*np.max(np.abs(self.features[feature]))
            xlim = (xmin, xmax)
            currentPlotId = 1
            for replicate in uniqueReplicates:
                plt.subplot(nrOfPlots, 1, currentPlotId)
                getIndices= np.where(self.replicateId == replicate)[0]
                fromIdx, toIdx = self.determineFromToIdx(np.arange(len(self.nrOfSamplesPerTissue)))
                useIdxToPlot = []
                for i in getIndices:
                    useIdxToPlot.append(np.arange(fromIdx[i], toIdx[i]))
                useIdxToPlot = np.concatenate(useIdxToPlot)
                title = "replicate "+str(replicate)
                self.plotFeature(feature, useIdxToPlot, title=title)
                plt.xlim(xlim)
                currentPlotId += 1
            plt.subplot(nrOfPlots, 1, currentPlotId)
            title = "all replicates pooled"
            self.plotFeature(feature, np.arange(np.sum(self.nrOfSamplesPerTissue)), title=title)
            plt.xlim(xlim)
            plt.legend(np.unique(self.labels))
            if self.baseFolder is None:
                plt.show()
            else:
                filename = self.baseFolder + self.perReplicateAppendix.format(feature)
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

    def plotFeaturesPerOverTimePoints(self, offset=0.4):
        self.nrOfReplicates = len(self.nrOfSamplesPerTissue) // self.nrOfTimepoints
        if self.seperateReplicates:
            if self.timePointDeclaration is None:
                nrOfPlots = self.nrOfTimepoints * self.nrOfReplicates
            else:
                nrOfPlots = len(self.timePointDeclaration)
        else:
            nrOfPlots = self.nrOfTimepoints
        for feature in self.featureNames:
            plt.subplots_adjust(hspace=0.7, wspace=0.7)
            xmin = np.min(self.features[feature])-offset*np.max(np.abs(self.features[feature]))
            xmax = np.max(self.features[feature])+offset*np.max(np.abs(self.features[feature]))
            xlim = (xmin, xmax)
            for timePoint in range(self.nrOfTimepoints):
                plt.subplot(nrOfPlots, 1, timePoint+1)
                idxOfTimePoint = self.determineIdxOfTimePoint(timePoint)
                if self.seperateReplicates:
                    for i in range(len(idxOfTimePoint)):
                        title = "{} at time step "+str(timePoint)+" P{}".format(i+1)
                        self.plotFeature(feature, idxOfTimePoint[i], title=title)
                        plt.xlim(xlim)
                else:
                    title = "{} at time step "+str(timePoint)
                    self.plotFeature(feature, idxOfTimePoint, title=title)
                    plt.xlim(xlim)
            plt.legend(np.unique(self.labels))
            if self.baseFolder is None:
                plt.show()
            else:
                filename = self.baseFolder + self.perTimePointAppendix.format(feature)
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

    def determineIdxOfTimePoint(self, timePoint):
        if self.timePointDeclaration is None:
            start = timePoint
            end = len(self.nrOfSamplesPerTissue) - self.nrOfTimepoints + timePoint
            idxOfReplicates = np.linspace(start, end, self.nrOfReplicates)
        else:
            idxOfReplicates = np.where(timePoint == self.timePointDeclaration)[0]
        fromIdx, toIdx = self.determineFromToIdx(idxOfReplicates)
        idxOfTimePoint = []
        for i in range(len(idxOfReplicates)):
            if self.seperateReplicates:
                idx = [np.arange(fromIdx[i], toIdx[i])]
            else:
                idx = np.arange(fromIdx[i], toIdx[i])
            idxOfTimePoint.append(idx)
        return np.concatenate(idxOfTimePoint)

    def determineFromToIdx(self, idxOfReplicates):
        fromIdx = []
        toIdx = []
        position = 0
        for i in range(len(self.nrOfSamplesPerTissue)):
            isIn = np.isin(i, idxOfReplicates)
            if isIn:
                fromIdx.append(position)
            position += self.nrOfSamplesPerTissue[i]
            if isIn:
                toIdx.append(position)
        return fromIdx, toIdx

    def determineCurrentAndNextPosition(self, nrOfSamplesPerTissue):
        nrOfSamples = len(nrOfSamplesPerTissue)
        allCurrentPos = np.zeros(nrOfSamples, dtype=int)
        allNextPos = np.zeros(nrOfSamples, dtype=int)
        currentPos = 0
        for i in range(nrOfSamples):
            allCurrentPos[i] = currentPos
            currentPos += nrOfSamplesPerTissue[i]
            allNextPos[i] = currentPos
        return allCurrentPos, allNextPos

    def PlotPooledData(self, featureIdx, color=None):
        if type(featureIdx) == str:
            featureIdx = np.where(self.featureNames == np.asarray(featureIdx))[0]
        data = self.features.iloc[:, featureIdx].values.flatten()
        sns.kdeplot(data, shade=True, color=color)#, bw="scott")

    def GetSelectedFeatures(self):
        return self.selectedFeatures

def getAllDensityPlotter(plotOnlyTesting, allFolders):
    allDensityPlotter = []
    for doOnlyTesting, folder in zip(plotOnlyTesting, allFolders):
        if doOnlyTesting:
            extension = "_test"
        else:
            extension = "_train"
        densityPlotter = mainSaveAllFeaturesTogetherFor(folder, extension,
                                                        filenameToSave=None,
                                                        doPlotAllFeaturesTogether=False)
        allDensityPlotter.append(densityPlotter)
    return allDensityPlotter

def combinePlotsPerRow(allFolders, plotOnlyTesting, filenameToSave, mode="single"):
    from pathlib import Path
    allDensityPlotter = getAllDensityPlotter(plotOnlyTesting, allFolders)
    allSelectedFeatures = [densityPlotter.GetSelectedFeatures() for densityPlotter in allDensityPlotter]
    lenOfSelectedFeatures = [len(selectedFeatures) for selectedFeatures in allSelectedFeatures]
    assert len(np.unique(lenOfSelectedFeatures)) == 1, "The number of selected features is different between the denisty plotter."
    selectedFeatures = allSelectedFeatures[0]
    nrOfSelectedFeatures = lenOfSelectedFeatures[0]
    print(filenameToSave)
    if mode == "simple":
        nrOfRows = np.ceil(np.sqrt(nrOfSelectedFeatures))
        plt.subplots_adjust(hspace=0.7, wspace=0.7)
        for i, currentFeature in enumerate(selectedFeatures):
            plt.subplot(nrOfRows, nrOfRows, i+1)
            for denistyPlotter in allDensityPlotter:
                denistyPlotter.PlotPooledData(currentFeature)
            plt.ylabel("density", fontsize = 15)
            plt.xlabel(currentFeature, fontsize = 15)
        # plt.legend(np.unique(self.labels))
    elif mode == "single":
        for i, currentFeature in enumerate(selectedFeatures):
            if i % 3 == 0:
                figsize = [3.15, 4.19]
            else:
                figsize = [3.15, 3.94]
            fig, ax = plt.subplots(figsize=figsize)
            for j, denistyPlotter in enumerate(allDensityPlotter):
                if j == 1:
                    color = None#"#ff5800"
                if j == 2:
                    color = None#"#d1008f"
                else:
                    color = None
                denistyPlotter.PlotPooledData(currentFeature, color=color)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            singlePlotName = Path(filenameToSave).parent.joinpath("singleFeatures", "{}_{}".format(currentFeature,Path(filenameToSave).name))
            plt.savefig(singlePlotName, dpi=200)
            plt.close()
    elif mode == "complex":
        nrOfRows = 2
        nrOfFeaturesPerWeighting = np.zeros(nrOfRows)
        mappingDictOfIdxToContainedString = {1:"topologyArea", 2:"topologyWall", 3:"topologyDist"}
        for currentFeature in selectedFeatures:
            idxToAddOne = 0
            for idx, stringToCheck in mappingDictOfIdxToContainedString.items():
                if stringToCheck in currentFeature:
                    idxToAddOne = idx
                    break
            nrOfFeaturesPerWeighting[idxToAddOne] += 1
        nrOfCols = np.max(nrOfFeaturesPerWeighting)
        nrOfPlotsPerWeight = np.zeros(nrOfRows)
        featureNameplotIdxAssociation = {}
        for currentFeature in selectedFeatures:
            idxToOfWeight = 0
            for idx, stringToCheck in mappingDictOfIdxToContainedString.items():
                if stringToCheck in currentFeature:
                    idxToOfWeight = idx
                    break
            plotIdx = int(1 + nrOfPlotsPerWeight[idxToOfWeight] + idxToOfWeight*nrOfCols)
            featureNameplotIdxAssociation[currentFeature] = plotIdx
            plt.subplot(nrOfRows, nrOfCols, plotIdx)
            for denistyPlotter in allDensityPlotter:
                denistyPlotter.PlotPooledData(currentFeature)
            if nrOfPlotsPerWeight[idxToOfWeight] == 0:
                plt.ylabel("density", fontsize = 15)
            plt.xlabel(currentFeature, fontsize = 15)
            nrOfPlotsPerWeight[idxToOfWeight] += 1
    if filenameToSave is None:
        plt.show()
    else:
        plt.savefig(filenameToSave, bbox_inches="tight", dpi=200, figsize=[78.7, 17.7])
        plt.close()
    if mode == "complex":
        pass
        # for i, currentFeature in enumerate(selectedFeatures):
        #     if "degree" in currentFeature:
        #         figsize = [3.15, 4.19]
        #     else:
        #         figsize = [3.15, 3.94]
        #     plt.figure(figsize=figsize)
        #     for denistyPlotter in allDensityPlotter:
        #         denistyPlotter.PlotPooledData(currentFeature)
        #     if "degree" in currentFeature:
        #         plt.ylabel("density", fontsize = 15)
        #         figsize = [3.15, 4.19]
        #     else:
        #         figsize = [3.15, 3.94]
        #     # xlabel = currentFeature
        #     # xlabel = xlabel.replace(" topologyArea", "")
        #     # xlabel = xlabel.replace(" topologyWall", "")
        #     # xlabel = xlabel.replace(" topologyDist", "")
        #     # xlabel = xlabel.replace(" on 2 neighborhood"," 2nd")
        #     # plt.xlabel(xlabel, fontsize = 15)
        #     singlePlotName = Path(filenameToSave).parent.joinpath("singleFeatures", "{}_{}".format(currentFeature,Path(filenameToSave).name))
        #     plt.savefig(singlePlotName, figsize=figsize, dpi=200)#, bbox_inches="tight"
        #     plt.close()
        # from PIL import Image
        # allImages = {}
        # for currentFeature in selectedFeatures:
        #     singlePlotName = Path(filenameToSave).parent.joinpath("singleFeatures", "{}_{}".format(currentFeature,Path(filenameToSave).name))
        #     image = Image.open(singlePlotName)
        #     image = image.convert("RGB")
        #     # print(currentFeature, image.size)
        #     allImages[currentFeature] = image
        # uniqueFeature = []
        # for currentFeature in selectedFeatures:
        #     if not "topologyArea" in currentFeature and not "topologyWall" in currentFeature and not "topologyDist" in currentFeature:
        #         uniqueFeature.append(currentFeature)
        # fullImageSize = [np.sum([allImages[f].size[0] for f in uniqueFeature]), 630*4+50*4]
        # fullImage = Image.new("RGB", fullImageSize, (255, 255, 255))
        # offset = [0, 0]
        # currentFeature = selectedFeatures[0]
        # currentImage = allImages[currentFeature]
        # fullImage.paste(currentImage, offset)
        # fullImage.paste(currentImage, currentImage.size)
        # plt.imshow(fullImage)
        # plt.show()
        # allImages = []
        # for currentFeature in selectedFeatures:
        #     singlePlotName = Path(filenameToSave).parent.joinpath("singleFeatures", "{}_{}".format(currentFeature,Path(filenameToSave).name))
        #     image = Image.open(singlePlotName)
        #     image = image.convert("RGB")
        #     allImages.append(mage)
        # image.save(Path(filenameToSave).parent.joinpath(Path(filenameToSave).name+".pdf"),
        #             save_all=True, append_images=allImages)

def mainSaveSameFeaturesOfDifferentHerkunftPerRow():
    allFolder = ["Results/divEventData/manualCentres/adjusted div Pred/allTopos/svm_k2h_combinedTable_l3f0n1c0ex0/",
                 "Results/divEventData/manualCentres/adjusted div Pred/allTopos/svm_k2h_combinedTable_l3f0n1c0ex0/",
                 "Results/ktnDivEventData/manualCentres/adjusted div Pred/allTopos/svm_k2h_combinedTable_l3f0n1c0ex0/"]
    plotOnlyTesting = [False, True, True]
    filenameToSave = "Results/Visualising Feature Distributions/comparingFeatureDistribution_allTopos_WTtrain_WTtest_ktntest.png"
    combinePlotsPerRow(allFolder, plotOnlyTesting, filenameToSave, mode="single")
    allFolder = ["Results/topoPredData/diff/manualCentres/bio/svm_k2h_combinedTable_l3f0n1c0ex1/",
                 "Results/topoPredData/diff/manualCentres/bio/svm_k2h_combinedTable_l3f0n1c0ex1/",
                 "Results/ktnTopoPredData/diff/manualCentres/bio/svm_k2h_combinedTable_l3f0n1c0ex1/"]
    plotOnlyTesting = [False, True, True]
    filenameToSave = allFolder[0] + "comparingFeatureDistribution_bio_WTtrain_WTtest_ktntest.png"
    combinePlotsPerRow(allFolder, plotOnlyTesting, filenameToSave, mode="single")

def mainSaveAllFeaturesTogetherFor(folder, extension, filenameToSave=None, doPlotAllFeaturesTogether=True):
    featureFilename = folder+"normalizedFeatures{}.csv".format(extension)
    labelVectorFilename = folder+"labels{}.csv".format(extension)
    baseFolder = None
    myFeatureDensityPlotting = FeatureDensityPlotting(featureFilename, labelVectorFilename,
                                                baseFolder=baseFolder)
    if doPlotAllFeaturesTogether:
        myFeatureDensityPlotting.plotAllFeaturesTogether(filenameToSave)
    return myFeatureDensityPlotting

def main():
    allFolder = ["Results/divEventData/manualCentres/adjusted div Pred/topoAndBio/svm_k2h_combinedTable_l3f0n1c0ex0/",
                 "Results/ktnDivEventData/manualCentres/adjusted div Pred/topoAndBio/svm_k2h_combinedTable_l3f0n1c0ex0/",
                 "Results/divEventData/manualCentres/adjusted div Pred/topoAndBio/svm_k2h_combinedTable_l3f0n1c0ex0/",
                 "Results/ktnDivEventData/manualCentres/adjusted div Pred/topoAndBio/svm_k2h_combinedTable_l3f0n1c0ex0/",
                 "Results/topoPredData/diff/manualCentres/bio/svm_k2h_combinedTable_l3f0n1c0ex1/",
                 "Results/ktnTopoPredData/diff/manualCentres/bio/svm_k2h_combinedTable_l3f0n1c0ex1/"]
    plotOnlyTesting = [False, True, False, True, False, True]
    allNamesToSave = ["topoAndBio_normalized_WT{}.png", "topoAndBio_normalized_ktn{}.png", "topoAndBio_normalized_WT{}.png", "topoAndBio_normalized_ktn{}.png", "bio_normalized_WT{}.png", "bio_normalized_ktn{}.png"]
    resultsFolder = "Results/Visualising Feature Distributions/"
    for doOnlyTesting, folder, nameToSave in zip(plotOnlyTesting, allFolder, allNamesToSave):
        if not doOnlyTesting:
            extension = "_train"
            filenameToSave = resultsFolder + nameToSave.format(extension)
            mainSaveAllFeaturesTogetherFor(folder, extension)#, filenameToSave)
        extension = "_test"
        filenameToSave = resultsFolder + nameToSave.format(extension)
        mainSaveAllFeaturesTogetherFor(folder, extension,)# filenameToSave)

if __name__ == '__main__':
    mainSaveSameFeaturesOfDifferentHerkunftPerRow()
    # main()
