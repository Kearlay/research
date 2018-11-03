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
#%%
# Load in libraries
from keras.models import load_model 
import time

import signal
import sys

import numpy as np

# Modules
from eeg_import import get_data, FNAMES
from eeg_preprocessing import prepare_data

#%%
# Path setting
filePath = "./data"
dataPath = filePath + "/test.edf"
modelPath = "../model"

# EEG info
sfreq = 128 #or 256

#%% (1)Load in data

def loadData(nth = 0):
    X,y = get_data([FNAMES[nth]], epoch_sec=0.0625)
    X_train, y_train, X_test, y_test = prepare_data(X, y)
    return X_train, y_train


#%% (3)Load in model
def loadModel(path = modelPath + "/CNN_full.h5"):
    model = load_model(path)
    return model

model = loadModel()

#%% (5) Predict - realtime
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
#%%
def printPred(model, interval):
    count = 0
    
    while True:
        X_train, y_train = loadData(count)
        count += 1
        print(f"working on for {count} times")
        with open('myPred.txt', 'a') as file:
            file.write("{:003}th record".format(count)+"\t"+\
                       str(model.predict(X_train[np.arange(10)*100]).argmax(axis=1)) +'\n')
        print(f'Please move for {interval} seconds')
        time.sleep(interval)

#%%
if __name__ == "__main__":
    mt = MyTimer().singleShot(total_running, sys.exit)        
    printPred(model, interval)

	

	



    

