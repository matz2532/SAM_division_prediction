import numpy as np
import pandas as pd
import re, sys
from pathlib import Path

def doZNormalise(array, useParameters=None, returnParameter=False):
    # z-score normalise the array along each column
    if not useParameters is None:
        mean = useParameters[0]
        std = useParameters[1]
    else:
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
    array = (array-mean)/std
    if returnParameter:
        return [array, [mean, std]]
    else:
        return array

def convertParentLabelingTableToDict(parentLabelingTable):
    parentLabelingDict = {}
    nrOfLabels = parentLabelingTable.shape[0]
    for i in range(nrOfLabels):
        daughter, parent = parentLabelingTable.iloc[i, 0], parentLabelingTable.iloc[i, 1]
        if parent in parentLabelingDict:
            parentLabelingDict[parent].append(daughter)
        else:
            parentLabelingDict[parent] = [daughter]
    return parentLabelingDict

class FooterExtractor (object):

    def __init__(self, fileToOpen=None, extractFooter=None,  saveToFilename=None):
        self.fileToOpen = fileToOpen
        self.extractFooter = extractFooter
        self.saveToFilename = saveToFilename
        self.lastLine = None
        self.AppendFooter()

    def AppendFooter(self, fileToOpen=None, extractFooter=None,  saveToFilename=None):
        self.lastLine = None
        if not fileToOpen is None:
            self.fileToOpen = fileToOpen
        if not extractFooter is None:
            self.extractFooter = extractFooter
        if not saveToFilename is None:
            self.saveToFilename = saveToFilename
        if not self.fileToOpen is None and not self.extractFooter is None:
            self.lastLine = self.extractLastLines(self.fileToOpen, self.extractFooter)
            if not self.saveToFilename is None:
                self.appendLinesToFile(self.lastLine, self.saveToFilename)

    def extractLastLines(self, fileToOpen, extractFooter=0):
        lastLines = ""
        file = open(fileToOpen, "r")
        allLines = file.readlines()
        file.close()
        if int(extractFooter) > 0:
            if len(allLines) >= extractFooter:
                lastLines = allLines[-extractFooter:]
                lastLines = "".join(lastLines)
                # do I need to change the range or the mesh number?
            else:
                print("The number of footer lines to extract ({}) was larger than the number of lines in the file {}.".format(extractFooter, fileToOpen))
        return lastLines

    def appendLinesToFile(self, lastLines, saveToFilename):
        file = open(str(saveToFilename), "a")
        file.write(lastLines)
        file.close()

    def GetLastLine(self):
        return self.lastLine

class convertTextToLabels (object):

    def __init__(self, textOrFilename, filenameToSave=None, allLabelsFilename=None,
                filenameToSaveControl=None, howToJoinLabels=", ", onlyUnique=True,
                warningsOn=False):
        self.howToJoinLabels = howToJoinLabels
        self.warningsOn = warningsOn
        self.lines = self.convertTextOrFilenameToLines(textOrFilename)
        self.labels = self.calculateLabels(self.lines)
        if not filenameToSave is None:
            self.saveLabels(filenameToSave, howToJoinLabels, onlyUnique)
        if not allLabelsFilename is None:
            if filenameToSaveControl is None:
                folder = Path(textOrFilename).parent
                filename = Path(textOrFilename).stem + "_control.csv"
                filenameToSaveControl = Path(folder).joinpath(filename)
            self.controlLabelPosition(filenameToSaveControl, allLabelsFilename, allLabelsAsHeatMapData=True)

    def convertTextOrFilenameToLines(self, textOrFilename):
        assert type(textOrFilename) is str, "You need to supply a string of text or a filename."
        splitByDot = textOrFilename.split(".")
        if len(splitByDot) > 1:
            if len(splitByDot) > 2:
                if self.warningsOn:
                    print("You may not passing a valid filename as you have more than one '.' .")
            file = open(textOrFilename, "r")
            textOrFilename = file.readlines()
            file.close()
            for i in range(len(textOrFilename)):
                textOrFilename[i] = textOrFilename[i].rstrip("\n")
        else:
            textOrFilename = textOrFilename.split("\n")
        return textOrFilename

    def calculateLabels(self, lines):
        labels = []
        for i in range(len(lines)):
            foundString = re.findall("\d+", lines[i])
            if len(foundString) > 0:
                if len(foundString) == 1:
                    labels.append(int(foundString[0]))
                else:
                    warningMsg = """More than one continious number was found in line {}: {}
Only the first number was added.""".format(i+1, foundString)
                    print(warningMsg)
                    labels.append(int(foundString[0]))
        if len(labels) == 0:
            print("No label was found as there were no integers found.")
        return labels

    def saveLabels(self, filenameToSave, howToJoinLabels, onlyUnique=True):
        labels = self.labels
        if onlyUnique is True:
            labels = np.unique(self.labels)
        joinedLabels = howToJoinLabels.join(labels)
        file = open(filenameToSave, "w")
        file.write(joinedLabels)
        file.close()

    def controlLabelPosition(self, saveToFilename, allLabels, allLabelsAsHeatMapData=True):
        labels = np.unique(self.labels)
        if allLabelsAsHeatMapData is True:
            table = pd.read_csv(allLabels, skipfooter=4, engine="python")
        else:
            print("other measures arent yet implemented. Set allLabelsAsHeatMapData to True")
            sys.exit()
        isLabelSelected = np.isin(table["Label"], labels)
        table["Value"] = np.asarray(isLabelSelected).astype(int)
        table.to_csv(saveToFilename, index=False)
        if allLabelsAsHeatMapData is True:
            lastLines = self.extractLastLines(allLabels, extractFooter=4)
            self.appendLinesToFile(lastLines, saveToFilename)

    def extractLastLines(self, fileToOpen, extractFooter=0):
        lastLines = ""
        file = open(fileToOpen, "r")
        allLines = file.readlines()
        file.close()
        if int(extractFooter) > 0:
            if len(allLines) >= extractFooter:
                lastLines = allLines[-extractFooter:]
                lastLines = "".join(lastLines)
                # do I need to change the range or the mesh number?
            else:
                print("The number of footer lines to extract ({}) was larger than the number of lines in the file {}.".format(extractFooter, fileToOpen))
        return lastLines

    def appendLinesToFile(self, lastLines, saveToFilename):
        file = open(str(saveToFilename), "a")
        file.write(lastLines)
        file.close()

    def GetLabels(self, onlyUnique=True):
        labels = np.asarray(self.labels).astype(int)
        if onlyUnique is True:
            labels = np.unique(labels)
        return labels

def main():
    plantNames = ["P2", "P5"]
    usePlantNamesAsFolder = True
    timePointsPerPlant = 5
    baseFolder = "Data/WT/"
    geometryText = "area"
    fileText = "dividingCells"
    saveExtension = "_extracted"
    labelFilename = "Data/WT/combinedAdditionalLabels_plusP1_centralRegion30_WeightedEdges.csv"
    labels = pd.read_csv(labelFilename)
    cellLabel = labels.iloc[:, 0]
    for i in range(len(plantNames)):
        if usePlantNamesAsFolder:
            extendedBaseFolder = baseFolder + plantNames[i] + "/"
        else:
            extendedBaseFolder = baseFolder
        for timePoint in range(timePointsPerPlant-1):
            sampleName = "{}T{}".format(plantNames[i], timePoint)
            allLabelsFilename = "{}{}{}.csv".format(extendedBaseFolder, geometryText, sampleName)
            sampleName += "T{}".format(timePoint+1)
            textOrFilename = "{}{}{}.txt".format(extendedBaseFolder, fileText, sampleName)
            filenameToSave = "{}{}{}{}.txt".format(extendedBaseFolder, fileText, sampleName, saveExtension)
            currentLabels = convertTextToLabels(textOrFilename, filenameToSave, allLabelsFilename).GetLabels()
            selectedCurrentLabels = []
            for j in range(len(currentLabels)):
                nrOfOccurence = np.sum(np.isin(cellLabel, currentLabels[j]))
                if nrOfOccurence == 1:
                    selectedCurrentLabels.append(j)
                elif nrOfOccurence > 1:
                    print("Label: {} was found {} times (comming from file {}), manual change is needed.".format(currentLabels[j], nrOfOccurence, textOrFilename))
            labelIdxToChange = np.where(np.isin(cellLabel, currentLabels[selectedCurrentLabels]))[0]
            labels.iloc[labelIdxToChange, 1] = 1
    labels.to_csv(labelFilename[:-4]+"_adjusted.csv", sep=",", index=False)

if __name__ == '__main__':
    main()
