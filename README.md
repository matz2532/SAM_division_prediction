## Topological properties accurately predict cell division events and organisation of Arabidopsis thaliana’s shoot apical meristem 
Timon W. Matz<sup>1,2</sup>, Yang Wang<sup>3</sup>, Ritika Kulshreshtha<sup>3</sup>, Arun Sampathkumar<sup>3</sup>, Zoran Nikoloski<sup>1,2</sup>

<sup>**1**</sup> Bioinformatics, Institute of Biochemistry and Biology, University of Potsdam,14476 Potsdam, Germany;

<sup>**2**</sup> Systems Biology and Mathematical Modelling, Max Planck Institute of Molecular Plant Physiology, 14476 Potsdam, Germany;

<sup>**3**</sup> Plant Cell Biology and Microscopy, Max Planck Institute of Molecular Plant Physiology, 14476 Potsdam, Germany
# Abstract
Cell division and the resulting changes to the cell organisation affect the shape and functionality of all tissues. Thus, understanding the determinants of the tissue-wide changes imposed by cell division is a key question in developmental biology. Here, we use a network representation of live cell imaging data from shoot apical meristems (SAMs) in Arabidopsis thaliana to predict cell division events and their consequences at a tissue level. We show that a classifier based on the SAM network properties is predictive of cell division events, with validation accuracy of 82% on par with that based on cell size alone. Further, we demonstrate that the combination of topological and biological properties, including: cell size, perimeter, distance, and shared cell wall between cells, can further boost the prediction accuracy of resulting changes in topology triggered by cell division. Using our classifiers, we demonstrate the importance of microtubule mediated cell to cell growth coordination in influencing tissue-level topology. Altogether the results from our network-based analysis demonstrates a feedback mechanism between tissue topology and cell division in A. thaliana’s SAMs.

# How to use the code
System requirements:

The code is written in python 3.8.1 using the following packages: matplotlib==3.1.3, networkx==2.4, numpy==1.18.1, pandas==1.0.3, seaborn==0.10.0, sklearn==0.22.1, scipy==1.4.1 

Either use anaconda or virtualenv to create a virtual environment with the specific python and package versions.
If you have not installed python, you can also install python with the packages versions directly on your computer.
