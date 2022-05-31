import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import pickle, sys, sklearn

from itertools import cycle
from numpy import interp
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import auc, plot_roc_curve, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.preprocessing import label_binarize

class MyScorer (object):

    f1Score=np.NaN
    accuracy=np.NaN
    precision=np.NaN
    TPRate=np.NaN
    FPRate=np.NaN
    auc=np.NaN
    sample_weight=np.NaN

    def __init__(self, y_true=None, y_pred=None, setAverage="not set", y_scores=None, nrOfClasses=2, weightLabelOccurence=True):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        self.nrOfClasses = nrOfClasses
        self.weightLabelOccurence = weightLabelOccurence
        isYTrueGiven = not self.y_true is None
        isYPredGiven = not self.y_pred is None
        if isYTrueGiven and isYPredGiven:
            self.calcAllScores(self.y_true, self.y_pred, setAverage=setAverage, y_scores=self.y_scores)

    def calcAllScores(self, y_true, y_pred, setAverage="not set", y_scores=None):
        if self.weightLabelOccurence:
            self.sample_weight = self.calcWeightLabelOccurence(y_true)
        else:
            self.sample_weight = None
        self.f1Score = self.calcF1Score(y_true, y_pred, setAverage=setAverage, checkSampleWeight=False)
        self.accuracy = self.calcAccuracy(y_true, y_pred, checkSampleWeight=False)
        self.precision = self.calcPrecision(y_true, y_pred, checkSampleWeight=False)
        if not y_scores is None:
            self.auc = self.calcAUC(y_true, y_scores, checkSampleWeight=False)

    def calcWeightLabelOccurence(self, y):
        n_samples = len(y)
        uniqueLabel, counts = np.unique(y, return_counts=True)
        n_classes = len(uniqueLabel)
        weightOfUniqueLabels = dict(zip(uniqueLabel, n_samples / (n_classes * counts)))
        sample_weight = np.zeros(n_samples)
        for label in uniqueLabel:
            isLabel = np.isin(y, label)
            sample_weight[isLabel.flatten()] = weightOfUniqueLabels[label]
        return sample_weight

    def calcF1Score(self, y_true, y_pred, inPercent=True, setAverage="not set", checkSampleWeight=True):
        if checkSampleWeight:
            if self.weightLabelOccurence:
                self.sample_weight = self.calcWeightLabelOccurence(y_true)
            else:
                self.sample_weight = None
        if setAverage == "not set":
            if self.nrOfClasses > 2:
                average = None
            else:
                average = "binary"
        else:
            average = setAverage
        f1Score = metrics.f1_score(y_true, y_pred, average=average, sample_weight=self.sample_weight)
        try:
            length = len(f1Score)
        except:
            length = self.nrOfClasses
        if length < self.nrOfClasses and average is None:
            f1Score = self.expandIndividualClassPerformanceOnMissingLabel(f1Score, y_true, scoreName="F1-score")
        if inPercent:
            return 100*f1Score
        else:
            return f1Score

    def expandIndividualClassPerformanceOnMissingLabel(self, perClassPerformances, expectedLabelsWithMissingClass, scoreName="F1-score"):
        uniqueFoundLabels = np.unique(expectedLabelsWithMissingClass)
        uniqueExpectedLabels = np.arange(self.nrOfClasses)
        isExpectedLabelFound = np.isin(uniqueFoundLabels, uniqueExpectedLabels)
        assert np.all(isExpectedLabelFound), f"While calculating the {scoreName}, there were fewer expected unique labels than the number of classes were given and\nthe labels: {uniqueFoundLabels[np.invert(isExpectedLabelFound)]} used are not in the expected format {uniqueExpectedLabels} (range(self.nrOfClasses))"
        newPerformanceScores = np.zeros(self.nrOfClasses)
        for i in range(len(perClassPerformances)):
            labelName = uniqueFoundLabels[i]
            idxOfLabelName = np.where(uniqueExpectedLabels == labelName)
            newPerformanceScores[idxOfLabelName] = perClassPerformances[i]
        return newPerformanceScores

    def calcAccuracy(self, y_true, y_pred, inPercent=True, checkSampleWeight=True):
        if checkSampleWeight:
            if self.weightLabelOccurence:
                self.sample_weight = self.calcWeightLabelOccurence(y_true)
            else:
                self.sample_weight = None
        accuracy = metrics.accuracy_score(y_true, y_pred, sample_weight=self.sample_weight)
        if inPercent:
            return 100*accuracy
        return accuracy

    def calcPrecision(self, y_true, y_pred, inPercent=True, setAverage="not set", checkSampleWeight=True):
        if checkSampleWeight:
            if self.weightLabelOccurence:
                self.sample_weight = self.calcWeightLabelOccurence(y_true)
            else:
                self.sample_weight = None
        if setAverage == "not set":
            if self.nrOfClasses > 2:
                average = None
            else:
                average = "binary"
        else:
            average = setAverage
        precision = metrics.precision_score(y_true, y_pred, average=average, sample_weight=self.sample_weight)
        try:
            length = len(precision)
        except:
            length = self.nrOfClasses
        if length < self.nrOfClasses and average is None:
            precision = self.expandIndividualClassPerformanceOnMissingLabel(precision, y_true, scoreName="precision")
        if inPercent:
            return 100*precision
        else:
            return precision

    def calcTruePositiveRate(self, y_true, y_pred, inPercent=True):
        confusionMatrix = self.calcTestedConfusionMatrix(y_true, y_pred)
        TPplusFN = np.sum(confusionMatrix[1,:])
        if TPplusFN > 0:
            truePositiveRate = confusionMatrix[1,1]/TPplusFN
        else:
            truePositiveRate = 0
        if inPercent:
            return 100*truePositiveRate
        else:
            return truePositiveRate

    def calcFalsePositiveRate(self, y_true, y_pred, inPercent=True):
        # do extensive testing
        confusionMatrix = self.calcTestedConfusionMatrix(y_true, y_pred)
        if self.nrOfClasses > 2:
            FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
            FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
            TP = np.diag(confusionMatrix)
            TN = confusionMatrix.sum() - (FP + FN + TP)
            falsePositiveRate = FP.astype(float)/(FP.astype(float)+TN.astype(float))
            if np.any(np.isin(falsePositiveRate, np.inf)):
                falsePositiveRate[np.isin(falsePositiveRate, inf)] = 0
            falsePositiveRate = np.mean(falsePositiveRate)
        else:
            TNplusFP = np.sum(confusionMatrix[0,:])
            if TNplusFP > 0:
                falsePositiveRate = confusionMatrix[0,1]/TNplusFP
            else:
                falsePositiveRate = 0
        if inPercent:
            return 100*falsePositiveRate
        else:
            return falsePositiveRate

    def calcTestedConfusionMatrix(self, y_true, y_pred):
        # I would suggest changing labels of confusion matrix here to adjust for multi classes
        confusionMatrix = metrics.confusion_matrix(y_true, y_pred)
        if len(confusionMatrix) < 2:
            if len(confusionMatrix) < 1:
                confusionMatrix = np.asarray([[0, 0], [0, 0]])
            else:
                if np.sign(y_true[0]) == +1:
                    confusionMatrix = np.asarray([[0, 0], [0, confusionMatrix[0][0]]])
                else:
                    confusionMatrix = np.asarray([[confusionMatrix[0][0], 0], [0, 0]])
        if len(confusionMatrix) > 2 and self.nrOfClasses <= 2:
            print("Warning: The confusion matrix detected more than 2 classes ({}), but the nrOfClasses is {}. {} > {}".format(len(confusionMatrix), self.nrOfClasses, len(confusionMatrix), self.nrOfClasses))
        return confusionMatrix

    def calcTrueNegativeRate(self, y_true, y_pred, inPercent=True):
        confusionMatrix = self.calcTestedConfusionMatrix(y_true, y_pred)
        if self.nrOfClasses > 2:
            FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)
            TP = np.diag(confusionMatrix)
            truePositiveRate = TP.astype(float)/(TN.astype(float)+FP.astype(float))
            if np.any(np.isin(truePositiveRate, inf)):
                truePositiveRate[np.isin(truePositiveRate, inf)] = 0
            truePositiveRate = np.mean(truePositiveRate)
        else:
            TNplusFP = np.sum(confusionMatrix[0,:])
            if TNplusFP > 0:
                truePositiveRate = confusionMatrix[0,0]/TNplusFP
            else:
                truePositiveRate = 0
        if inPercent:
            return 100*truePositiveRate
        else:
            return truePositiveRate

    def calcAUC(self, y_true, y_scores, checkSampleWeight=True):
        if checkSampleWeight:
            if self.weightLabelOccurence:
                self.sample_weight = self.calcWeightLabelOccurence(y_true)
            else:
                self.sample_weight = None
        if self.nrOfClasses == 3:
            binarized_y_true = label_binarize(y_true, classes=[0, 1, 2])
            sample_weight = self.calcWeightLabelOccurence(binarized_y_true.ravel())
            fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), y_scores.ravel(),
                                sample_weight=sample_weight,
                                drop_intermediate=True)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_scores,
                                sample_weight=self.sample_weight,
                                drop_intermediate=True)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def GetAllScores(self, y_true=None, y_pred=None, setAverage="not set", y_scores=None,
                     return1DList=False):
        isYTrueGiven = not y_true is None
        isYPredGiven = not y_pred is None
        if isYTrueGiven and isYPredGiven:
            self.calcAllScores(y_true, y_pred, setAverage=setAverage, y_scores=y_scores)
        elif self.y_true is None or self.y_pred is None:
            assert False, "You need to give y_true and y_pred in GetAllScores() or while object initialisation."
        performance = [self.f1Score, self.accuracy, self.precision]
        if return1DList:
            if type(performance[0]) == np.ndarray:
                for f1 in performance[0][::-1]:
                    performance.insert(1, f1)
                performance[0] = np.mean(performance[0])
        if not y_scores is None or not self.y_scores is None:
            performance.append(self.auc)
        return performance

    def GetAllScoresFromConfusionMatrix(self, confusionMatrix):
        accuracy = 100*(confusionMatrix[0,0]+confusionMatrix[1,1])/np.sum(confusionMatrix)
        f1Score = 100*2*confusionMatrix[1,1]/(2*confusionMatrix[1,1]+confusionMatrix[1,0]+confusionMatrix[0,1])
        TPRate = 100*confusionMatrix[1,1]/(confusionMatrix[1,1]+confusionMatrix[1,0])
        TNRate = 100*confusionMatrix[0,0]/(confusionMatrix[0,0]+confusionMatrix[0,1])
        return f1Score, accuracy, TPRate, TNRate

    def CalculateRocCurve(self, y_true, y_prob, checkSampleWeight=True):
        if checkSampleWeight:
            if self.weightLabelOccurence:
                self.sample_weight = self.calcWeightLabelOccurence(y_true)
            else:
                self.sample_weight = None
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, sample_weight=self.sample_weight)
        else:
            self.calcMultiClassOrcCurve(y_true, y_prob, n_classes)
        return fpr, tpr, thresholds

    def calcMultiClassOrcCurve(self, y_true, y_prob, n_classes):
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_true[:, i], y_prob[:, i], sample_weight=self.sample_weight)
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
        return fpr, tpr, thresholds

    def SetNrOfClasses(self, nrOfClasses):
        self.nrOfClasses = nrOfClasses

def plotRocCurveWithCv(modelsOfSplit, valXs, valYs, n_classes=3, plotShow=True,
                       title="diff features + concat normPerTissue hyper rbf svm",
                       plotLegend=True):
    import copy
    means = []
    stds = []
    mean_fpr = np.linspace(0, 1, 1000)
    lineColors = ['aqua', 'darkorange', 'deeppink']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    classAucs, classTprs = calcAucAndTprForEachClass(modelsOfSplit, valXs, valYs, n_classes, mean_fpr)

    meanAucs, meanTprs = calcMeanAucAndTprs(modelsOfSplit, valXs, valYs, n_classes, mean_fpr)
    mean_tpr = np.mean(meanTprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(meanAucs)
    means.append(mean_auc)
    stds.append(std_auc)
    lw=2
    label = 'average ROC curve (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    label = "All classes"
    plt.plot(mean_fpr, mean_tpr,
             label=label,
             color='cornflowerblue', linewidth=lw+1)#, linestyle=':'
    std_tpr = np.std(meanTprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="cornflowerblue", alpha=.5)#, label=r'$\pm$ 1 std. dev.')

    for i in range(n_classes):
        tprs = classTprs[i]
        aucs = classAucs[i]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        means.append(mean_auc)
        stds.append(std_auc)
        label = r'Mean ROC of class %i (AUC = %0.2f $\pm$ %0.2f)' % (i, mean_auc, std_auc)
        label = "Class {}".format(i)
        ax.plot(mean_fpr, mean_tpr, color=lineColors[i],
                label=label,
                lw=lw, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=lineColors[i], alpha=.3)#, label=r'$\pm$ 1 std. dev.')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',
            label='Chance', alpha=.8)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=title)
    if plotLegend:
        ax.legend(loc="lower right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if plotShow:
        plt.show()
    return means, stds

def calcAucAndTprForEachClass(modelsOfSplit, valXs, valYs, n_classes,
                                 mean_fpr=np.linspace(0, 1, 100)):
    setPosLabel = 1
    classTprs = []
    classAucs = []
    for pos_label in range(n_classes):
        scoresToCombine = np.delete(np.arange(n_classes), pos_label)
        tprs = []
        aucs = []
        for splitNr in range(len(modelsOfSplit)):
            if hasattr(modelsOfSplit[splitNr], "best_estimator_"):
                model = modelsOfSplit[splitNr].best_estimator_
            else:
                model = modelsOfSplit[splitNr]
            X = valXs[splitNr]
            y_true = copy.deepcopy(valYs[splitNr])
            y_scores = model.decision_function(X)
            isPosLabel = np.isin(y_true, pos_label)
            y_true[isPosLabel] = setPosLabel
            y_true[np.invert(isPosLabel)] = 0
            newY_scores = np.zeros((y_scores.shape[0], 2))
            newY_scores[:, 0] = y_scores[:, pos_label]
            newY_scores[:, 1] = np.sum(y_scores[:, scoresToCombine], axis=1)
            newY_scores = scipy.special.softmax(newY_scores, axis=1)
            newY_scores = newY_scores[:, 0]
            fpr, tpr, _ = roc_curve(y_true, newY_scores, pos_label=setPosLabel,
                                sample_weight=None,
                                drop_intermediate=True)
            roc_auc = auc(fpr, tpr)
            viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=None)
            # pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        classTprs.append(tprs)
        classAucs.append(aucs)
    return classAucs, classTprs

def calcMeanAucAndTprs(modelsOfSplit, valXs, valYs, n_classes, mean_fpr):
    setPosLabel = 1
    meanTprs = []
    meanAucs = []
    for splitNr in range(len(modelsOfSplit)):
        if hasattr(modelsOfSplit[splitNr], "best_estimator_"):
            model = modelsOfSplit[splitNr].best_estimator_
        else:
            model = modelsOfSplit[splitNr]
        X = valXs[splitNr]
        y_true = copy.deepcopy(valYs[splitNr])
        y_true = label_binarize(y_true, classes=[0, 1, 2])
        y_scores = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_scores.ravel(),
                            sample_weight=None,
                            drop_intermediate=True)
        roc_auc = auc(fpr, tpr)
        viz = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=None)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        meanTprs.append(interp_tpr)
        meanAucs.append(viz.roc_auc)
    return meanAucs, meanTprs

def plotRocCurve(y_true, y_scores, n_classes=3, plotShow=True, splitNr=1):
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    y_true = label_binarize(y_true, classes=[0, 1, 2])
    np.random.seed(12)
    for i in range(n_classes):
        selectedSamples = np.isin(y_true, 1)
        zeroClassIdx = np.where(np.invert(selectedSamples))[0]
        idxToSelect = np.random.randint(0, len(zeroClassIdx), len(zeroClassIdx)//2)
        zeroClassIdx = zeroClassIdx[idxToSelect]
        selectedSamples[zeroClassIdx] = True
        selectedSamples = np.where(selectedSamples)[0]
        selectedSamples = np.arange(len(y_true))
        selectedYTrue, selectedYScores = y_true[selectedSamples, i], y_scores[selectedSamples, i]
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(selectedYTrue, selectedYScores)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='cornflowerblue', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve of split {}'.format(splitNr))
    plt.legend(loc="lower right")
    if plotShow:
        plt.show()

def plotPrecisionRecallCurve(y_true, y_scores, n_classes=3, plotShow=True, splitNr=1):
    precision = dict()
    recall = dict()
    thresholds = dict()
    precisionRecallAuc = dict()
    y_true = label_binarize(y_true, classes=[0, 1, 2])
    np.random.seed(12)
    for i in range(n_classes):
        selectedSamples = np.isin(y_true, 1)
        zeroClassIdx = np.where(np.invert(selectedSamples))[0]
        idxToSelect = np.random.randint(0, len(zeroClassIdx), len(zeroClassIdx)//2)
        zeroClassIdx = zeroClassIdx[idxToSelect]
        selectedSamples[zeroClassIdx] = True
        selectedSamples = np.where(selectedSamples)[0]
        selectedSamples = np.arange(len(y_true))
        selectedYTrue, selectedYScores = y_true[selectedSamples, i], y_scores[selectedSamples, i]
        precision[i], recall[i], thresholds[i] = metrics.precision_recall_curve(selectedYTrue, selectedYScores)
        precisionRecallAuc[i] = metrics.average_precision_score(y_true[:, i], y_scores[:, i])
    # Compute micro-average ROC curve and ROC area+
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(), y_scores.ravel())
    precisionRecallAuc["micro"] = metrics.average_precision_score(y_true.ravel(), y_scores.ravel())
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(precisionRecallAuc["micro"]),
             color='cornflowerblue', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                 ''.format(i, precisionRecallAuc[i]))
    # plt.plot([0, 1], [0.5, 0.5], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curve of split {}'.format(splitNr))
    plt.legend()
    if plotShow:
        plt.show()

def makePerformanceHumanReadable(performanceValues, allFeatureSetNames):
    nrOfSets = len(allFeatureSetNames)
    nrOfClasses = len(performanceValues)//nrOfSets
    assert nrOfSets*nrOfClasses == len(performanceValues), f"The performanceValues seem not to have the correct number of entries, {len(performanceValues)} != {nrOfSets*nrOfClasses}={nrOfSets}*{nrOfClasses}"
    performanceDict = {}
    i = 0
    for featureName in allFeatureSetNames:
        classPerformanceDict = {}
        classPerformanceDict["avg"] = performanceValues[i]
        i += 1
        for classNr in np.arange(nrOfClasses-1):
            classPerformanceDict[classNr] = performanceValues[i]
            i += 1
        performanceDict[featureName] = classPerformanceDict
    return performanceDict

def plotGivenFeatureSetRocCurves(resultsBaseFolder="Results/topoPredData/diff/manualCentres/",
                                 modelBaseFolder="Results/topoPredData/diff/manualCentres/",
                                 excludeDivNeighbours=True, saveFig=True, useValidationData=True,
                                 featureSets=["allTopos", "bio", "topoAndBio"],
                                 saveUnderFolder=None, doStdOnTestTissues=True):
    if saveUnderFolder is None:
        saveUnderFolder = resultsBaseFolder
    allMeans, allStds = [], []
    if excludeDivNeighbours:
        excludingTxt = "ex1"
    else:
        excludingTxt = "ex0"
    for i, set in enumerate(featureSets):
        folderExtension = "svm_k1h_combinedTable_l3f0n1c0bal0{}/".format(excludingTxt)
        resultsFolder = resultsBaseFolder + "{}/{}".format(set, folderExtension)
        modelFolder = modelBaseFolder + "{}/{}".format(set, folderExtension)
        title = ""
        if useValidationData:
            dataName = "val"
            modelName = "models.pkl"
            modelsOfSplit = pickle.load(open(modelFolder + modelName, "rb"))
        else:
            dataName = "test"
            modelName = "testModel.pkl"
            modelsOfSplit = [pickle.load(open(modelFolder + modelName, "rb"))]
        dataXs = pickle.load(open(resultsFolder + "{}Xs.pkl".format(dataName), "rb"))
        labelYs = pickle.load(open(resultsFolder + "{}Ys.pkl".format(dataName), "rb"))
        if not useValidationData and doStdOnTestTissues:
            import pandas as pd
            labelOverviewDf = pd.read_csv(resultsFolder+"labelOverviewDf.csv")
            labelOverviewDf = labelOverviewDf.loc[labelOverviewDf["isCellTest"], :]
            groups = labelOverviewDf.groupby(["plant", "time point"])
            nrOfTissues = 0
            newDataXs, newLabelYs = [], []
            startIdx = 0
            for tissueId, tissueOverview in groups:
                nrOfSamples = len(tissueOverview)
                endIdx = startIdx + nrOfSamples
                newDataXs.append(dataXs[0][startIdx:endIdx, :])
                newLabelYs.append(labelYs[0][startIdx:endIdx])
                nrOfTissues += 1
                startIdx = endIdx
            modelsOfSplit = nrOfTissues * [modelsOfSplit[0]]
            dataXs = newDataXs
            labelYs = newLabelYs
        plotLegend = i == 0
        means, stds = plotRocCurveWithCv(modelsOfSplit, dataXs, labelYs, title=title, plotShow=False,
                           plotLegend=plotLegend)
        allMeans.extend(list(means))
        allStds.extend(list(stds))
        filenameToSave = saveUnderFolder + "Roc-Curves/"
        Path(filenameToSave).mkdir(parents=True, exist_ok=True)
        filenameToSave += "roc curve {} {}.png".format(excludingTxt, set)
        if saveFig:
            plt.tight_layout()
            plt.savefig(filenameToSave)
        plt.close()
    aucMeanDict = makePerformanceHumanReadable(allMeans, featureSets)
    aucStdDict = makePerformanceHumanReadable(allStds, featureSets)
    print("allMeans", aucMeanDict)
    print("allStds", aucStdDict)
    return allMeans, allStds

def mainPlotRocCurvesAndAUCLabelDetails(resultsBaseFolder="",
                modelBaseFolder=None, useValidationData=False,
                saveUnderFolder=None):
    if modelBaseFolder is None:
        modelBaseFolder = resultsBaseFolder
    means, stds = plotGivenFeatureSetRocCurves(saveFig=True, resultsBaseFolder=resultsBaseFolder,
                                               modelBaseFolder=modelBaseFolder,
                                               useValidationData=useValidationData,
                                               saveUnderFolder=saveUnderFolder)
    import sys
    sys.path.insert(0, "./Code/Analysis Tools/")
    from BarPlotPlotter import mainTopoPredRandomization
    doSpecial = [means, stds]
    mainTopoPredRandomization(performance="AUC", doSpecial=doSpecial, doMainFig=False,
                              baseResultsFolder=resultsBaseFolder, savePlotFolder=saveUnderFolder)

def main():
    mainPlotRocCurvesAndAUCLabelDetails(resultsBaseFolder="Results/topoPredData/diff/manualCentres/")
    mainPlotRocCurvesAndAUCLabelDetails(resultsBaseFolder="Results/ktnTopoPredData/diff/manualCentres/",
                                        modelBaseFolder="Results/topoPredData/diff/manualCentres/")
    sys.exit()
    # folder = "Results/divEventData/area/svm_k1h_combinedTable_l3f0n1c0ex0/"
    # folder = "Results/Tissue mapping/reducedFeatures_d26/onlyTopoDistWithParent/svm_k1h_combinedTable_l3f0n1c0ex0/"
    # folder = "Temporary/topoAndBio/svm_k1h_combinedTable_l3f0n1c0ex1/"
    # idx = 1
    # plotGivenFeatureSetRocCurves(resultsBaseFolder="Temporary/", featureSets=["topoAndBio"], saveFig=True)
    # model = pickle.load(open(folder+"models.pkl", "rb"))[idx]
    # valXs = pickle.load(open(folder+"valXs.pkl", "rb"))[idx]
    # valYs = pickle.load(open(folder+"valYs.pkl", "rb"))[idx]
    # y_scores = model.decision_function(valXs)
    # y_pred = model.predict(valXs)
    # auc = MyScorer(nrOfClasses=len(np.unique(valYs))).GetAllScores(valYs, y_pred=y_pred, y_scores=y_scores)
    # print(auc)
    # sys.exit()
    weightedF1Scores = []
    individualF1Scores = []
    nrOfSplits = len(modelsOfSplit)
    print("nr of splits: {}".format(nrOfSplits))
    for splitNr in range(nrOfSplits):
        if hasattr(modelsOfSplit[splitNr], "best_estimator_"):
            model = modelsOfSplit[splitNr].best_estimator_
        else:
            model = modelsOfSplit[splitNr]
        X = valXs[splitNr]
        y_true = valYs[splitNr]
        y_scores = model.decision_function(X)
        y_pred = model.predict(X)
        individualF1Scores.append(MyScorer().calcF1Score(y_true, y_pred))
        # plotRocCurve(y_true, y_scores, splitNr=splitNr+1, plotShow=True)
        # plotPrecisionRecallCurve(y_true, y_scores, splitNr=splitNr+1, plotShow=True)
        weightedF1Scores.append(MyScorer().calcF1Score(y_true, y_pred, setAverage="weighted"))
        # filenameToSave = resultsFolder + "roc-curve "
        # plt.savefig(, bbox_inches="tight")
        print(MyScorer().calcAccuracy(y_true, y_pred))
    print("{}%{}%".format(np.round(np.mean(weightedF1Scores), 2), np.round(np.std(weightedF1Scores), 2)))
    print(np.concatenate(individualF1Scores).reshape(nrOfSplits, 3))
    print(np.mean(individualF1Scores, axis=0))
    print(np.std(individualF1Scores, axis=0))
    # myMyScorer = MyScorer()
    # myMyScorer.CalculateRocCurve(y_true, y_prob)
    # print(myMyScorer.calcF1Score(y_true, y_pred))
    # print(myMyScorer.GetAllScoresFromConfusionMatrix(confusionMatrix))

if __name__ == '__main__':
    main()

# y_prob = scipy.special.softmax(y_scores, axis=1)
# macro_roc_auc_ovo = metrics.roc_auc_score(y_true, y_prob, multi_class="ovo",
#                                   average="macro")
# weighted_roc_auc_ovo = metrics.roc_auc_score(y_true, y_prob, multi_class="ovo",
#                                      average="weighted")
# macro_roc_auc_ovr = metrics.roc_auc_score(y_true, y_prob, multi_class="ovr",
#                                   average="macro")
# weighted_roc_auc_ovr = metrics.roc_auc_score(y_true, y_prob, multi_class="ovr",
#                                      average="weighted")
# print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
# print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#       "(weighted by prevalence)"
#       .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

#       mean                  0                   1                   2
# 0.7505164516658769, 0.7558008374100328, 0.8018635406566441, 0.6706891583328365,
# 0.7281482806387021, 0.6623922250359031, 0.6443927731284054, 0.8266252930046034,
# 0.8183693358980715, 0.8027558906869252, 0.8218625626671605, 0.8188569970179165
