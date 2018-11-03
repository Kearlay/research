'''
Author: Euiyoung Chung
Data: Oct. 10, 2018
Descriptions:

This script is writted to test the model trained by 'eeg_main.py' for new real time eeg data.
Device: Emotiv Epoc +
Number of channel: 14
Sampling rate: 128 or 256
Frequency range: 0.16 - 43Hz

'''

# Load in libraries
from mne.io import read_raw_edf
from mne import pick_types

from keras.models import load_model 
from sklearn.preprocessing import scale

import numpy as np
import time

import signal
import sys

from collections import defaultdict

# Path setting
filePath = "./data"
dataPath = filePath + "/test.edf"
modelPath = "./model"

# EEG info
sfreq = 128 #or 256

# (1)Load in data
def loadData(interval, dataPath = dataPath):
    raw = read_raw_edf(dataPath, preload=True, verbose=False)
    eegPicks = pick_types(raw.info, eeg=True)
    raw.filter(l_freq=1, h_freq=None, picks=eegPicks)
    return raw.get_data(picks=eegPicks)[-sfreq*interval:,:]

# (2)Preprocess data - 2D meshes + normalization + sliding windows 
# Assume the input array shape [None, 14]
# Reshape the input into [None, 10, 6, 6,1] - [batch, window, width, height, 1]

def preprocess(data):
    
    def scaler(data):
        ''' The data should be shaped as
        [None, 14] 
        '''
        return scale(data, axis=1)
        
    def slidingWindow(data):
        
        n_window = data.shape[0]//10 - 1
        n_frame = 10
        n_channel = 14
        
        newData = np.zeros((n_window, n_frame, n_channel))

        for i in range(n_window):
            newData[i, :, :] = data[5*i:5*i+10, :]
        
        return newData
            
    def mesh(data):
        n_window = data.shape[0]
        n_frame = 10
        height = width = 6
        
        newData = np.zeros((n_window, n_frame, height, width))
        
        newData[:, :, 0, np.array([1, 4])] =
        newData[:, :, 1, np.array([0,2,3,5])] =
        newData[:, :, 2, np.array([1, 4])] =
        newData[:, :, 4, np.array([0, 5])] =
        newData[:, :, 5, np.array([1, 4])] = 
        
        return newData
        
    return mesh(slidingWindow(scaler(data)))

# (3)Load in model
def loadModel(path = modelPath + "/model.h5"):
    model = load_model(path)
    return model

model = loadModel()

# (4)Predict - assuming we know the true values
'''
if __name__ == "__main__":
	y_pred = []	
	for i in testData.shape[0]:
	 	y_pred.append(model.predict(test).argmax(axis = 1))

	score = classification_report(y_test, y_pred, digits=4, output_dict=True)
	print(score)
'''

# (5) Predict - realtime
total_running = int(sys.argv[2]) #seconds
interval = int(sys.argv[1]) #seconds

class MyTimer(object):
    """
    https://pythonadventures.wordpress.com/2012/12/08/terminate-a-script-after-x-seconds/
    Similar to Qt's QTimer. Call a function in a specified time.
 
    Time is given in sec. Usage:
    mt = MyTimer()
    mt.singleShot(<sec>, <function_name>)
 
    After setting it, you can still disable it:
    mt.disable()
 
    If you call it several times, any previously scheduled alarm
    will be canceled (only one alarm can be scheduled at any time).
    """
    def singleShot(self, sec, func):
        self.f = func
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(sec)
 
    def handler(self, *args):
        self.f()
 
    def disable(self):
        signal.alarm(0)

def printPred(model, interval):
    
    while True:
        data = loadData(interval)
        data = preprocess(data)

        print(model.predict(data).argmax(axis=1))

        time.sleep(interval)


if __name__ == "__main__":
    mt = MyTimer().singleShot(total_running, sys.exit)        
    printPred(model, interval)

	

	



    

