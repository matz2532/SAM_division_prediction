import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn
import sys

modulePath = "./Code/Classifiers/"
sys.path.insert(0, modulePath)
from NestedModelCreator import NestedModelCreator
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

def plot_coefficients(classifier, featureNames, topFeatures=None,
                      filenameToSave=None, showPlot=True, useBothDirectionTopFeatures=True,
                      highColor="red", lowColor="blue", addFeatureNamesToPlot=False,
                      printOut=True, addEmptyInMiddle=False):
    coef = classifier.coef_.ravel()
    coefficientSorting = np.argsort(coef)
    if not topFeatures is None:
        if useBothDirectionTopFeatures:
            topPositiveCoefficients = coefficientSorting[-topFeatures:]
            topNegativeCoefficients = coefficientSorting[:topFeatures]
            coefficientSorting = np.hstack([topNegativeCoefficients, topPositiveCoefficients])
        else:
            coefficientSorting = np.argsort(np.abs(coef))[-topFeatures:]
    # create plot
    sortedCoef = coef[coefficientSorting]
    featureNames = np.array(featureNames)
    sortedFeatureNames = featureNames[coefficientSorting]
    if addEmptyInMiddle:
        sortedCoef = list(sortedCoef)
        sortedCoef.insert(len(sortedCoef)//2, 0)
        sortedCoef.insert(len(sortedCoef)//2, 0)
        sortedCoef = np.asarray(sortedCoef)
        sortedFeatureNames = list(sortedFeatureNames)
        sortedFeatureNames.insert(len(sortedFeatureNames)//2, "")
        sortedFeatureNames.insert(len(sortedFeatureNames)//2, "")
        sortedFeatureNames = np.asarray(sortedFeatureNames)
    colors = [highColor if c < 0 else lowColor for c in sortedCoef]
    nrOfUsedCoef = len(sortedCoef)
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    plt.rcParams.update({'font.size': 20})
    if addFeatureNamesToPlot:
        plt.figure(figsize=(8, 5))
    else:
        plt.figure(figsize=(5, 5))
    plt.barh(np.arange(nrOfUsedCoef), sortedCoef, color=colors)
    ticks = np.arange(nrOfUsedCoef)
    if addFeatureNamesToPlot:
        usedFeatureNames = sortedFeatureNames
        # if addEmptyInMiddle:
        #     idxToIgnore = len(featureNames)//2
        #     ticks = list(ticks)
        #     ticks.pop(idxToIgnore)
        #     usedFeatureNames = list(usedFeatureNames)
        #     usedFeatureNames.pop(idxToIgnore)
        plt.yticks(ticks, usedFeatureNames)#, rotation=60, ha="right")
    else:
        plt.yticks(ticks, [""]*nrOfUsedCoef)
    plt.xlim((-3, 3))
    plt.xticks([-2, -1, 0, 1, 2])
    sortedFeatureNamesTxt = "\n".join(sortedFeatureNames[::-1])
    if showPlot and filenameToSave is None:
        if not addFeatureNamesToPlot:
            print(sortedFeatureNamesTxt)
        plt.tight_layout()
        plt.show()
    elif not filenameToSave is None:
        if not topFeatures is None:
            filenameToSave = Path(filenameToSave).with_name(Path(filenameToSave).stem + f"_topFeatures{topFeatures}" + Path(filenameToSave).suffix)
        if not addFeatureNamesToPlot:
            with open(Path(filenameToSave).with_name(Path(filenameToSave).stem + "_featureNames.txt"), "w") as fh:
                fh.write(sortedFeatureNamesTxt)
        plt.savefig(filenameToSave, bbox_inches="tight")
        plt.close()
        if printOut:
            print(f"saved {filenameToSave}")

#load model
topFeatures = 5
resultsFolder = "Results/divEventData/manualCentres/lowCor0.3/svm_k1h_combinedTable_l3f0n1c0bal0ex0/"
features = pd.read_csv(resultsFolder + "normalizedFeatures_test.csv")
filenameToSave = resultsFolder + "featureImportance.png"
# filenameToSave = None

with open(resultsFolder + "testModel.pkl", "rb") as fh:
    svm = pickle.load(fh)
svm = svm.GetModel().best_estimator_

featureNames = np.asarray(list(features.columns))
# featureNames = [i.replace("centrality", "c") for i in featureNames]
# featureNames = [i.replace("on 2 neighborhood", "2nd n") for i in featureNames]
colorPallette = seaborn.color_palette("colorblind")
highColor = colorPallette[3]
lowColor = colorPallette[0]
plot_coefficients(svm, featureNames, topFeatures=topFeatures, filenameToSave=filenameToSave,
                  highColor=highColor, lowColor=lowColor)
