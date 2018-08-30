# System
import requests
import re
import os
import pathlib
import urllib

# Modeling & Preprocessing
import keras.layers as layers
from keras.models import Sequential, model_from_json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras import initializers, optimizers

# Essential Data Handling
import numpy as np
import pandas as pd

# Get Paths
from glob import glob

# EEG package
from mne import pick_types
from mne.io import read_raw_edf

# Modules
from eeg_import import get_data
from eeg_preprocessing import prepare_data

#%%

# Get file paths
PATH = '/Users/jimmy/data/PhysioNet/'
SUBS = glob(PATH + 'S[0-9]*')
FNAMES = sorted([x[-4:] for x in SUBS])

# Remove subject #89 with damaged data
FNAMES.remove('S089')

#%%
X,y = get_data(FNAMES, epoch_sec=0.0625)

print(X.shape)
print(y.shape)

#%%

X_train, y_train, X_test, y_test = prepare_data(X, y)

# Check out the shape of the mesh
np.set_printoptions(precision=2, linewidth=100)
X_train[1][0]
#%%

# Make another dimension, 1, to apply CNN for each time frame.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

## Complicated Model - the same as Zheng`s
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
lecun = initializers.lecun_normal(seed=42)

#CNN
CNN = Sequential()
CNN.add(layers.Conv2D(32, (3,3), padding='same', activation='elu',\
                      data_format='channels_last',kernel_initializer=lecun))
CNN.add(layers.Conv2D(64, (3,3), padding='same', activation='elu',\
                      data_format='channels_last',kernel_initializer=lecun))
CNN.add(layers.Conv2D(128, (3,3), padding='same', activation='elu',\
                      data_format='channels_last',kernel_initializer=lecun))
CNN.add(layers.Flatten())
CNN.add(layers.Dense(1024, activation='elu', kernel_initializer=lecun))
CNN.add(layers.Dropout(0.5))

#RNN
model = Sequential()
model.add(layers.TimeDistributed(CNN, input_shape=input_shape))
model.add(layers.LSTM(64, return_sequences=True, kernel_initializer=lecun))
model.add(layers.LSTM(64,kernel_initializer=lecun))
model.add(layers.Dense(1024, activation='elu', kernel_initializer=lecun))
CNN.add(layers.Dropout(0.5))

model.add(layers.Dense(5, activation='softmax'))

model.summary()

def sd_pred(y_true, y_pred):
    return K.std(y_pred)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001), metrics=['accuracy', sd_pred])
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=50)