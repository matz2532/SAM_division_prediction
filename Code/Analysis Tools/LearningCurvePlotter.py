import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as st
import sys

class LearningCurvePlotter (object):

    def __init__(self, testPerformance, trainPerformance, nrOfSamples,
                yLabel="F1-score", showPlot=True,
                trainLabel = "Training score", testLabel = "Test score",
                useConfidenceIntervall=False, limitYAxis=False):
        self.yLabel = yLabel
        self.useConfidenceIntervall = useConfidenceIntervall
        self.trainLabel = trainLabel
        self.testLabel = testLabel
        self.limitYAxis = limitYAxis
        self.plot_learning_curve(nrOfSamples, testPerformance, trainPerformance)
        if showPlot:
            plt.show()

    def plot_learning_curve(self, nrOfTrainingSamples, testPerformance, trainPerformance,
                            ylim=None):
        plt.figure()
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Number of training samples")
        plt.ylabel(self.yLabel)
        if self.limitYAxis:
            ylim = [np.min([np.min(testPerformance), np.min(trainPerformance)]),
                    np.max([np.max(testPerformance), np.max(trainPerformance)])]
        else:
            ylim = [0, 100]
        self.plotPerformance(nrOfTrainingSamples, trainPerformance,
                             label=self.trainLabel, alpha=0.1, color="r")
        self.plotPerformance(nrOfTrainingSamples, testPerformance,
                             label=self.testLabel, alpha=0.1, color="g")
        plt.ylim(ylim[0], ylim[1])
        plt.legend(loc="lower right")
        return plt

    def plotPerformance(self, nrOfTrainingSamples, performanceMatrix, label, alpha=0.1, color="r"):
        performanceMatrix = performanceMatrix.astype(float)
        performance_mean = np.mean(performanceMatrix, axis=1)
        if self.useConfidenceIntervall:
            confidenceIntervall = st.t.interval(0.95, len(performanceMatrix)-1, loc=np.mean(performanceMatrix, axis=1), scale=st.sem(performanceMatrix, axis=1))
            self.plotFillBetween(nrOfTrainingSamples, confidenceIntervall[0],
                             confidenceIntervall[1], alpha=alpha,
                             color=color)
        else:
            performance_std = np.std(performanceMatrix, axis=1)
            self.plotFillBetween(nrOfTrainingSamples, performance_mean - performance_std,
                             performance_mean + performance_std, alpha=alpha,
                             color=color)
        plt.plot(nrOfTrainingSamples, performance_mean, '-', color=color,
                 label=label)

    def plotFillBetween(self, x, y1, y2, color="g", alpha=0.1):
        plt.fill_between(x, y1, y2, alpha=alpha, color=color)

def selectPerformence(allTestPerformence, allTrainPerformence, selectedPerfromenece):
    testPerformence, trainPerformence = [], []
    for i in range(len(allTestPerformence)):
        testPerformence.append(list(allTestPerformence[i][:, selectedPerfromenece]))
        trainPerformence.append(list(allTrainPerformence[i][:, selectedPerfromenece]))
    return testPerformence, trainPerformence

def main():
    selectedPerformance = 1
    performanceModi = ['f1-score', 'accuracy', 'TP rate', 'TN rate']
    folder = "Results/streamLined/42/r30_notWeightedEdges_zNormTissueWise_removedTissue_8_13_14_19/testing on P1/"
    filenameNrOfSamples = folder + "nrOfLabels_centralRegion30_WeightedEdges_linear.pkl"
    filenameTestPerfromence = folder + "allTest['f1-score', 'accuracy', 'TP rate', 'TN rate']_centralRegion30_WeightedEdges_linear.pkl"
    filenameTrainPerfromence = folder + "allTraining['f1-score', 'accuracy', 'TP rate', 'TN rate']_centralRegion30_WeightedEdges_linear.pkl"
    nrOfSamples = pickle.load(open(filenameNrOfSamples, "rb"))
    testPerformance = pickle.load(open(filenameTestPerfromence, "rb"))
    trainPerformance = pickle.load(open(filenameTrainPerfromence, "rb"))
    testPerformance, trainPerformance = selectPerformence(testPerformance, trainPerformance, selectedPerformance)
    myLearningCurvePlotter = LearningCurvePlotter(testPerformance, trainPerformance, nrOfSamples) # , yLabel=performanceModi[selectedPerformance]

if __name__ == '__main__':
    main()
