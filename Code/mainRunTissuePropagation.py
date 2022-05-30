import sys

sys.path.insert(0, "./Code/Propagation/")

from DivAndTopoPredictor import propagateAndCorrelateTissues
from LocalTopologyPredictionComparer import compareLocalTopologyPrediction
from mainAnalyseOccurrenceAndPerformance import plotAndPrepareMainFigures
from RandomisedTissuePrediction import mainRandomiseWithMultiplePlants

def main():
    # print("Do calculations for figure 4C")
    # compareLocalTopologyPrediction()
    # plotAndPrepareMainFigures(figuresToDo="Fig. 4 C")
    # print("Do calculation for figure D (predicted vs observed), E, and Supp Fig 8")
    # propagateAndCorrelateTissues()
    # plotAndPrepareMainFigures(figuresToDo="Fig. 4 E") # and Sup. Fig. 8
    print("Do calculation for figure D (random vs observed)")
    mainRandomiseWithMultiplePlants()
    plotAndPrepareMainFigures(figuresToDo="Fig. 4 D")

if __name__== "__main__":
    main()
