
Brain Signal Anaylsis Projects
==========

This repository contains groups of scripts written as a part of EEG classification research for 
the state-of-the-art deep learning application.

## Columbia Data Science Institute (DSI) Conference
*"Brain State Classification on Functional Near Infrared Spectroscopy using Convolutional Neural Networks"* <br>Andrew Lee, Jim (Euiyoung) Chung, Xiaofu He

 The main project in progress. Our lab is planning finalize the paper soon and the source code will be updated after the paper is published.<br>
![Poster](https://github.com/Kearlay/research/blob/master/conference0928.png?raw=true)

Overview
--------

TensorFlow and Keras implementation of Zhang et al(2018), "EEG-based Intention Recognition from Spatio-Temporal Representations via Cascade and Parallel Convolutional Recurrent Neural Networks" for EEG motar imagery classification on PhysioNet data (https://www.physionet.org/pn4/eegmmidb/). Stacked CNN and RNN were applied on time-distributed sliding windows of raw EEG data.


Acamdemic Reference
------------
directory | title                                               |  author                             |        year
--------|-------------------------------------------------------|-------------------------------------|-----------------
BCI4 | *Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface* | Kai Keng Ang, et al. | 2008
BCI4 | *Filter bank common spatial pattern algorithm on BCI competition IV Datasets 2a and 2b* | Kai Keng Ang, et al.               | 2012
BCI4 | *Feature Selection for Motor Imagery EEG Classification Based on Firefly Algorithm and Learning Automata* | Aiming Liu, et al. | 2017
deepLearning  | *Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks* | Pouya Bashivan, et al. | 2016
deepLearning  | *EEG-Based Emotion Recognition Using Deep Learning Network with Principal Component Based Covariate Shift Adaptation* | Suwicha Jirayucharoensak, et al. | 2014
deepLearning  | *A Deep Learning Architecture for Temporal Sleep Stage Classification Using Multivariate and Multimodal Time Series* | Stanislas Chambon, et al. | 2018
deepLearning  | *Deep Learning With Convolutional Neural Networks for EEG Decoding and Visualization* |Robin T. Schirrmeister, et al.   | 2017
deepLearning     | *EEG-based Intention Recognition from Spatio-Temporal Representations via Cascade and Parallel Convolutional Recurrent Neural Networks* | Dalin Zhang, et al. | 2018
deepLearning | *A Deep Learning Method for Classification of EEG Data Based on Motor Imagery* | Xiu An, et al. | 2014
emotionState | *Classifying Different Emotional States by Means of EEG- Based Functional Connectivity Patterns* | You-Yun Lee, Shulan Hsieh | 2014
emotionState | *Emotion Classification Based on Gamma-band EEG* | Mu Li, Bao-Liang Lu | 2009
preprocessing | *Time-series discrimination using feature relevance analysis in motor imagery classification* | A.M. Alvarez-Meza, et al. | 2014
preprocessing | *Comparison of signal decomposition methods in classification of EEG signals for motor-imagery BCI system*              | Jasmin Kevric, et al. | 2015
preprocessing | *Classification of EEG Motor imagery multi class signals based on Cross Correlation* | D.Hari Krishna, et al.                | 2016
preprocessing | *EEG Signal Processing Techniques For Mental Task Classification* | Rajveer Shastri, et al. | 2015



Description of files
--------------------

- All code is written in python3
- Dependencies include MNE (EEG handling package), Keras, requests, urllib3
- Python scripts are adjusted to be run on HPC

Non-Python files:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
README.md                         |  Text file (markdown format) description of the project.

Python Scripts:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
eeg_main.py                       | Import EEG data from data files and start training the graph (added for HPC clusters)
eeg_tensorflow.ipynb              | TensorFlow and Keras implementation. Please refer to this notebook for a quick look

Python Modules (HPC version):

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
eeg_preprocessing.py              | This contains useful preprocessing steps to implement spatio-temporal pattern recognition on raw eeg data. Based on Scikit-learn and MNE pacakges.
eeg_import.py                     | Functions defined to extract data from .edf file format using MNE package.
eeg_data_downloads.py             | Executing this code will generate folders and start downloading PhysioNet data into them.
eeg_eval.py                       | Evaluation of the model based on the history file. - calls the confusion matrix, loss, and accuracy.
eeg_prepare.py                    | Preprocess the imported EEG data. 
gpu_training.sh                   | Send the eeg_main.py script to the computational nodes. Enter 'sbatch gpu_trainin.sh' on Habanero HPC.


