import numpy as np
import pandas as pd

class StandardTableFormater (object):

    def __init__(self, plantNames=None, currentPlantName=None, currentTimePoint=None):
        self.plantNames = plantNames
        self.currentPlantName = currentPlantName
        self.currentTimePoint = currentTimePoint

    def SetPlantNames(self, plantNames):
        self.plantNames = plantNames

    def SetPlantNameByName(self, plantName):
        self.currentPlantName = plantName

    def SetPlantNameByIdx(self, plantIdx):
        assert not self.plantNames is None, "self.plantNames is not set yet. self.plantNames = {}".format(self.plantNames)
        self.currentPlantName = self.plantNames[plantIdx]

    def SetTimePoint(self, timePoint):
        self.currentTimePoint = timePoint

    def GetProperStandardFormatWith(self, array, returnList=True):
        assert not self.currentPlantName is None, "The current plant name is not set yet. self.currentPlantName = {}".format(self.currentPlantName)
        assert not self.currentTimePoint is None, "The current time point is not set yet. self.currentTimePoint = {}".format(self.currentTimePoint)
        if returnList:
            outList = [self.currentPlantName, self.currentTimePoint]
            for entry in array:
                outList.append(entry)
            return outList
        else:
            return np.concatenate([[self.currentPlantName], [self.currentTimePoint], array])

    def GetFormatForMappedCells(self, mappedCells, withColumns=None,
                                returnOnlyUnique=False, addKeyCell=False,
                                addMappedDaughter=False):
        sourceOfCells = []
        alreadySeenLabel = []
        for keyCell, mappedNeighbors in mappedCells.items():
            baseSourceOfCells = [self.currentPlantName,
                                 self.currentTimePoint]
            if addKeyCell:
                baseSourceOfCells.append(keyCell)
            for mappedParent, mappedDaughter in mappedNeighbors.items():
                currentSourceOfCells = baseSourceOfCells.copy()
                currentSourceOfCells.append(mappedParent)
                if addMappedDaughter:
                    currentSourceOfCells.append(mappedDaughter)
                if not mappedParent in alreadySeenLabel or not returnOnlyUnique:
                    sourceOfCells.append(currentSourceOfCells)
                    if returnOnlyUnique:
                        alreadySeenLabel.append(mappedParent)
        if withColumns is None:
            return sourceOfCells
        else:
            table = pd.DataFrame(sourceOfCells, columns=withColumns)
            return table

    def GetProperStandardFormatWithEveryEntryIn(self, array, extend=False):
        assert not self.currentPlantName is None, "The current plant name is not set yet. self.currentPlantName = {}".format(self.currentPlantName)
        assert not self.currentTimePoint is None, "The current time point is not set yet. self.currentTimePoint = {}".format(self.currentTimePoint)
        fullOutList = []
        for entry in array:
            outList = [self.currentPlantName, self.currentTimePoint]
            if extend:
                outList.extend(entry)
            else:
                outList.append(entry)
            fullOutList.append(outList)
        return fullOutList

def main():
    myArray = np.arange(5)
    myStandardTableFormater = StandardTableFormater(["P1", "P2"])
    myStandardTableFormater.SetTimePoint("2")
    myStandardTableFormater.SetPlantNameByIdx(1)
    # out = myStandardTableFormater.GetProperStandardFormatWith(myArray)
    # import pandas as pd
    # table = pd.DataFrame(out).T
    # for i in table.iloc[0,:]:
    #     print(type(i))
    # print(out)
    predictedMapping = {3: {221: 514, 78: 516, 9: 483, 143: 653, 7: 504, 130: 701}, 240: {408: 763.0, 409: 554.0, 233: 534.0, 215: 602.0, 45: 500.0, 128: 490.0}, 236: {135: 568, 78: 516, 103: 685, 112: 710, 33: 559}, 339: {307: 764, 348: 470, 350: 619, 106: 620, 82: 586, 325: 766}, 415: {242: 553, 86: 758, 430: 540, 396: 521, 377: 505, 361: 636, 105: 666, 416: 694}, 334: {60: 765, 107: 647, 407: 650, 363: 599, 296: 769, 39: 691, 196: 686}, 337: {361: 615, 377: 505, 91: 634, 336: 614, 302: 751, 323: 478}, 348: {306: 674, 307: 764, 339: 584, 350: 619, 381: 638, 379: 637, 80: 489, 324: 469}, 135: {236: 776, 78: 516, 9: 483, 147: 625, 4: 693, 33: 559}, 359: {413: 708, 103: 633, 335: 580, 346: 611, 99: 503}, 78: {221: 775, 413: 708, 103: 685, 236: 776, 135: 568, 9: 483, 3: 592}, 335: {359: 632, 346: 611, 319: 570, 62: 545, 336: 614, 103: 633}, 413: {412: 728, 221: 514, 78: 549, 103: 519, 359: 511, 99: 684}, 302: {337: 597, 336: 477, 62: 747, 274: 563, 47: 768, 275: 785, 305: 670, 323: 571}, 11: {429: 486, 155: 527, 313: 572, 27: 631, 42: 658, 260: 657}, 45: {264: 659.0, 95: 561.0, 65: 700.0, 215: 602.0, 240: 575.0, 128: 490.0}, 204: {407: 650.0, 363: 599.0, 366: 491.0, 220: 716.0, 136: 582.0, 35: 518.0}, 112: {236: 732, 103: 633, 336: 614, 91: 634, 104: 664, 33: 559}, 99: {413: 708, 412: 728, 375: 661, 346: 611, 359: 511}, 106: {383: 640.0, 82: 586.0, 339: 601.0, 350: 619.0, 398: 654.0, 13: 715.0, 162: 695.0}, 229: {157: 485, 153: 482, 220: 716, 366: 491, 338: 773, 260: 735, 74: 777, 34: 509}, 361: {337: 488, 377: 505, 415: 665, 105: 666, 362: 617, 71: 581, 323: 478}, 221: {412: 728, 413: 708, 78: 516, 3: 592, 130: 701}, 63: {275: 785, 305: 670, 323: 478, 71: 581, 80: 489, 324: 469, 306: 674, 277: 770, 40: 756, 259: 790}, 42: {74: 777.0, 341: 541.0, 27: 631.0, 11: 596.0, 260: 657.0}, 149: {202: 656, 429: 486, 86: 758, 4: 778, 147: 625, 257: 524}, 4: {430: 540, 86: 758, 149: 626, 147: 625, 135: 593, 33: 734}, 409: {418: 671, 419: 673, 283: 531, 233: 534, 240: 720, 408: 763, 162: 695}, 128: {45: 500, 240: 575, 408: 763, 394: 761, 180: 564, 27: 598, 264: 659}, 336: {335: 580.0, 62: 747.0, 302: 660.0, 337: 597.0, 91: 634.0, 112: 652.0, 103: 633.0}, 429: {202: 656, 149: 467, 86: 779, 155: 527, 11: 629, 260: 657}, 27: {264: 659, 262: 698, 341: 541, 42: 658, 11: 596, 313: 572, 180: 564, 128: 490}, 86: {430: 540, 415: 712, 242: 553, 155: 527, 429: 486, 149: 467, 4: 778}, 398: {431: 713, 6: 714, 13: 715, 106: 620, 350: 619, 381: 638, 417: 668}, 33: {430: 540, 396: 687, 104: 664, 112: 710, 236: 776, 135: 568, 4: 778}, 323: {361: 615, 337: 488, 302: 751, 305: 670, 63: 759, 71: 581}, 296: {147: 655, 39: 691, 334: 487, 363: 599, 72: 772, 257: 524}, 260: {338: 773, 202: 656, 429: 486, 11: 629, 42: 697, 74: 777, 229: 492}, 147: {296: 731.0, 39: 691.0, 9: 623.0, 135: 593.0, 4: 693.0, 149: 626.0, 257: 524.0}, 396: {430: 540, 415: 665, 377: 505, 91: 634, 104: 664, 33: 734}, 346: {375: 651, 333: 494, 319: 544, 335: 580, 359: 632, 99: 503}, 9: {143: 653, 196: 686, 39: 691, 147: 625, 135: 568, 78: 516, 3: 592}}
    columnNames = ["plant", "time point", "parent neighbor"]
    out2 = myStandardTableFormater.GetFormatForMappedCells(predictedMapping, columnNames, True)
    print(out2)
    print(np.unique(out2.iloc[:,2], return_counts=True)[1])

if __name__ == '__main__':
    main()
