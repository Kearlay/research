
Brain Signal Anaylsis Projects
==========

This repository contains groups of scripts written as a part of EEG classification research for 
the state-of-the-art deep learning application. Collaborated with researchers at Columbia University Medical Center.

Overview
--------

TensorFlow API, Keras implementation of Zhang et al(2018), "EEG-based Intention Recognition from Spatio-Temporal Representations via Cascade and Parallel Convolutional Recurrent Neural Networks" for EEG motar imagery classification on PhysioNet data (https://www.physionet.org/pn4/eegmmidb/). Stacked CNN and RNN were applied on time-distributed sliding windows of raw EEG data.


Acamdemic Papers
------------

project | title                                                 |  author                             |        year
--------|-------------------------------------------------------|-------------------------------------|-----------------
Med/Vis | Computer Vision in Healthcare Applications            | Junfeng Gao, et al.                 | 2017
Med     | Deep Learning in Medical Image Analysis               | Dinggang Shen, et al.               | 2017
Med     | Deep Learning in Medical Imaging: General Overview    | June-Goo Lee, et al.                | 2017
Med     | Hello World Deep Learning in Medical Imaging          | Paras Lakhani                       | 2018
Med     | Overview of deep learning in medical imaging          | Kenji Suzuki                        | 2017
Med     | Automated analysis of retinal imaging using machine learning techniques for computer vision | De Fauw, J. et al.                        | 2017
Med     | Blood type classification using computer vision and machine learning | Ana Ferraz, et al.   | 2015
Med     | 3D computer vision based on machine learning with deep neural networks: A review            | Kailas Vodrahalli, et al.                       | 2017
Vis/Med | Learning the Image Processing Pipeline                | Haomiao Jiang, et al.               | 2017
Vis/Med | scikit-image: image processing in Python              | Ste ́fan van der Walt                | 2014


Description of files
--------------------

- All code is written in python3

Non-Python files:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
README.md                         |  Text file (markdown format) description of the project.

Python scripts files:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
fetch_oeis_database.py            |  Fetch and refresh data from the remote OEIS database to a local sqlite3 database.

Python modules:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
fraction_based_linear_algebra.py  |  Perform matrix inversion without loss of precision using the Fraction type.



# eeg_main.py

Keras implementation of Zhang et al(2018), "EEG-based Intention Recognition from Spatio-Temporal Representations via Cascade and Parallel Convolutional Recurrent Neural Networks" for EEG motar imagery classification on PhysioNet data (https://www.physionet.org/pn4/eegmmidb/). Stacked CNN and RNN were applied on time-distributed sliding windows of raw EEG data.

# eeg_preprocessing.py

This contains useful preprocessing steps to implement spatio-temporal pattern recognition on raw eeg data. Based on Scikit-learn and MNE pacakges.

# eeg_import.py

Functions defined to extract data from .edf file format using MNE package.

# eeg_data_downloads.py

Executing this code will generate folders and start downloading PhysioNet data into them.

# eeg.ipython

Jupyter notebook style

# Reference 

For the convenience of reading, I collected some basic and important papers about EEG processing.
