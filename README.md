For fNIRS and EEG research at Columbia

# Reference

For the convenience of reading, I collected some basic and important papers about EEG processing.

# eeg_main.py

Keras implementation of Zhang et al(2018), "EEG-based Intention Recognition from Spatio-Temporal Representations via Cascade and Parallel Convolutional Recurrent Neural Networks" for EEG motar imagery classification on PhysioNet data (https://www.physionet.org/pn4/eegmmidb/). Stacked CNN and RNN were applied on time-distributed sliding windows of raw EEG data.

# eeg_preprocessing.py

This contains useful preprocessing steps to implement spatio-temporal pattern recognition on raw eeg data. Based on Scikit-learn and MNE pacakges.

# eeg_data_downloads.py

Executing this code will generate folders and start downloading PhysioNet data into them.

# eeg.ipython

Jupyter notebook style
