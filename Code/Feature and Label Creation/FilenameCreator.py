class FilenameCreator (object):

    def __init__(self, baseFolder, timePointsPerPlant, plantNames,
                usePlantNamesAsFolder=True, lengthOfTimeStep=1,
                addlastTimePoint=False,
                connectivityText="cellularConnectivityNetwork",
                parentLabelingText="parentLabeling", geometryText="area",
                peripheralLabelText="periphery labels "):
        self.plantNames = plantNames
        self.usePlantNamesAsFolder = usePlantNamesAsFolder
        self.addlastTimePoint = addlastTimePoint
        self.connectivityText = connectivityText
        self.parentLabelingText = parentLabelingText
        self.geometryText = geometryText
        self.peripheralLabelText = peripheralLabelText
        self.setFilenames(baseFolder, timePointsPerPlant, lengthOfTimeStep=lengthOfTimeStep)

    def setFilenames(self, baseFolder, timePointsPerPlant, lengthOfTimeStep=1):
        assert lengthOfTimeStep > 0, "lengthOfTimeStep needs to be positive. It is {}".format(lengthOfTimeStep)
        self.connectivityNetworkFilenames = []
        self.parentLabelingFilenames = []
        self.areaFilenames = []
        self.peripheralLabelsFilename = []
        extendedBaseFolder = []
        for i in range(len(self.plantNames)):
            if self.usePlantNamesAsFolder:
                extendedBaseFolder.append(baseFolder + self.plantNames[i] + "/")
            else:
                extendedBaseFolder.append(baseFolder)
        if self.addlastTimePoint:
            selectedTimePoints = timePointsPerPlant
        else:
            selectedTimePoints = timePointsPerPlant-1
        for timePoint in range(selectedTimePoints):
            for i in range(len(self.plantNames)):
                sampleName = "{}T{}".format(self.plantNames[i], timePoint)
                filename = "{}{}{}.csv".format(extendedBaseFolder[i], self.connectivityText, sampleName)
                self.connectivityNetworkFilenames.append(filename)
                filename = "{}{}{}.csv".format(extendedBaseFolder[i], self.geometryText, sampleName)
                self.areaFilenames.append(filename)
                filename = "{}{}{}.txt".format(extendedBaseFolder[i], self.peripheralLabelText, sampleName)
                self.peripheralLabelsFilename.append(filename)
                sampleName += "T{}".format(timePoint + lengthOfTimeStep)
                filename = "{}{}{}.csv".format(extendedBaseFolder[i], self.parentLabelingText, sampleName)
                self.parentLabelingFilenames.append(filename)

    def GetConnectivityNetworkFilenames(self):
        return self.connectivityNetworkFilenames

    def GetAreaFilenames(self):
        return self.areaFilenames

    def GetPeripheralLabelsFilenames(self):
        return self.peripheralLabelsFilename

    def GetPartentLabellingFilenames(self):
        return self.parentLabelingFilenames

def main():
    baseFolder = "./Data/WT/"
    plantNames = ["P1","P2", "P5", "P6", "P8"]
    myFilenameCreator = FilenameCreator(baseFolder, 5, plantNames)
    connectivityNetworkFilenames = myFilenameCreator.GetConnectivityNetworkFilenames()
    areaFilenames = myFilenameCreator.GetAreaFilenames()
    peripheralLabelsFilename = myFilenameCreator.GetPeripheralLabelsFilenames()
    parentLabelingFilenames = myFilenameCreator.GetPartentLabellingFilenames()
    print(connectivityNetworkFilenames)
    print(areaFilenames)
    print(peripheralLabelsFilename)
    print(parentLabelingFilenames)

if __name__ == '__main__':
  main()
