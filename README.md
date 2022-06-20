## Topological properties accurately predict cell division events and organisation of Arabidopsis thaliana’s shoot apical meristem
Timon W. Matz<sup>1,2</sup>, Yang Wang<sup>3</sup>, Ritika Kulshreshtha<sup>3</sup>, Arun Sampathkumar<sup>3</sup>, Zoran Nikoloski<sup>1,2</sup>

<sup>**1**</sup> Bioinformatics, Institute of Biochemistry and Biology, University of Potsdam,14476 Potsdam, Germany;

<sup>**2**</sup> Systems Biology and Mathematical Modelling, Max Planck Institute of Molecular Plant Physiology, 14476 Potsdam, Germany;

<sup>**3**</sup> Plant Cell Biology and Microscopy, Max Planck Institute of Molecular Plant Physiology, 14476 Potsdam, Germany
# Abstract
Cell division and the resulting changes to the cell organization affect the shape and functionality of all tissues. Thus, understanding the determinants of the tissue-wide changes imposed by cell division is a key question in developmental biology. Here, we use a network representation of live cell imaging data from shoot apical meristems (SAMs) in Arabidopsis thaliana to predict cell division events and their consequences at a tissue level. We show that a support vector machine classifier based on the SAM network properties is predictive of cell division events, with test accuracy of 76%, matching that based on cell size alone. Further, we demonstrate that the combination of topological and biological properties, including: cell size, perimeter, distance, and shared cell wall between cells, can further boost the prediction accuracy of resulting changes in topology triggered by cell division. Using our classifiers, we demonstrate the importance of microtubule mediated cell-to-cell growth coordination in influencing tissue-level topology. Together, the results from our network-based analysis demonstrate a feedback mechanism between tissue topology and cell division in A. thaliana’s SAMs.

# How to use the code
System requirements:

The code is written in python 3.8.1 using the following packages: matplotlib==3.1.3, networkx==2.4, numpy==1.18.1, pandas==1.0.3, seaborn==0.10.0, sklearn==0.22.1, scipy==1.4.1

Either use anaconda or virtualenv to create a virtual environment with the specific python and package versions.
If you have not installed python, you can also install python with the packages versions directly on your computer.

How to run the code (top-down perspective):

Execute the code from the base folder (so that you are in the README.md file).
1. Run mainConvertRawDataToFeaturesAndLabels.py to convert the table information (see below for how to extract the table information from MGX surface) into features and labels for division event and local topology prediction of WT and ktn SAM (24h) and floral meristem (12h time steps) plants.
2. Run mainTrainValTestWT.py to train and validate (per plant 6-fold cross-validation) SVMs having linear kernel on WT SAM features and labels (including learning curve generation); test data on model (retrained on train-validation data; test model); save performance measures in results folder 'Results/\[divEventData, topoPredData/diff\]/manualCentres/{feature set}/svm_k1h_combinedTable_l3f0n1c0e\[ex0, ex1\]/resultsWithTesting.csv' (where \[a, b\] either a or b is insterted and the feature set may be: 'allTopos', 'area', 'bio', 'topoAndBio', 'lowCor0.3', 'topology'
3. Run mainTestKtn.py to test ktn SAM data on the test model trained on WT data and saved in 'Results/ktn\[DivEvent, TopoPred\]Data/manualCentres/{feature set}/svm_k1h_combinedTable_l3f0n1c0e\[ex0, ex1\]resultsWithTesting.csv'.
4. Run mainTestFloralMeristems.py to test WT and ktn floral meristem data on the test model trained on WT SAM data and saved in 'Results/floral meristems/\[WT, ktn\]/\[DivEvent, TopoPred\]Data/manualCentres/{feature set}/svm_k1h_combinedTable_l3f0n1c0e\[ex0, ex1\]resultsWithTesting.csv'.
5. Run mainRunTissuePropagation.py run WT SAM propagation, saving results under 'Results/DivAndTopoApplication/' for all tissue time steps of P2 and P9 (N=8 tissue time steps).
6. Run mainAnalyseOccurrenceAndPerformance.py to save summarised results i.e., the main figures, and supplemental figures and tables.

Needed table information:

The data is/needs to be structured in the 'Data' folder. The WT and ktn SAM and floral meristem data is separated in further folders. The individual replicates are saved as P1, P2 or ktnP1, ktnP2 (for WT and ktn, respectively).
You need to extract and save the tables in the replicate folder
1. the topology table from MGX (MGX function: Mesh/Export/Save Cell Neighborhood 2D) and save it under the name 'cellularConnectivityNetworkP**x**T**y**.csv', where **x** represents the replicates id and **y** is the time point id.
2. the geometry table from MGX (MGX: Mesh/Heat Map/Heat Map Classic) and save it under the name 'areaP**x**T**y**.csv' and 'areaktnP**x**T**y**.csv' (for WT and ktn, respectively).
3. a parent labeling file, where you connect the parent cells with the daughter cells tracking the cell lineage. You have to do the lineage tracking for all cells possible linking, e.g. T0 with T1 and name it 'fullParentLabelingP**x**T**y**T**y+1**.csv' or 'fullParentLabelingktnP**x**T**y**T**y+1**.csv' (for WT and ktn, respectively)
4. select all cells at the "edge" of the tissue separated by a new line 'periphery labelsP**x**T**y**T**y+1**.txt' and 'periphery labels ktnP**x**T**y**T**y+1**.txt' (for WT and ktn, respectively). For this you can use the pipet tool of MGX, simultaneously press Ctrl + Alt + Left mouse button one after another selecting all peripheral cells and copy the text of the MGX terminal and paste it in the text file.
