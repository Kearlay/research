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

# Modules
from eeg_import_test import get_data_forTesting, FNAMES #you have to pip3 install mne which will load the .edf raw eeg data and also preproessing
from eeg_preprocessing_test import prepare_data_test #also use mne package

# Warnings
import warnings
warnings.filterwarnings('always')
"""
#%%
# Path setting
filePath = "./data"
dataPath = filePath + "/test.edf"
modelPath = "../model"
"""

# Path setting, defined in the eeg_import.py
#filePath = "..\data\BCI2000\S001" #E:\hxf\Faculty\ColumbiaUniversity\research\myLab\RAs\EEG\Euiyoung(Jimmy)Chung\project\EEG\Demo\data\BCI2000\S001
#dataPath = filePath + "\S001R04.edf"
modelPath = "../model"


# EEG info
sfreq = 128 #or 256
windowTime=0.0625 #sliding window time
#%% (1)Load in data

def loadData(which_subject, c, interval): #return the 1st subject if nth=0, 7 runs, run#2,4,6,8,10,12,14 refer to E:\hxf\Faculty\ColumbiaUniversity\research\myLab\RAs\EEG\Euiyoung(Jimmy)Chung\project\EEG\Demo\data\BCI2000
    X,y = get_data_forTesting([FNAMES[which_subject]], c, interval=interval, epoch_sec=windowTime) #FNAMES and get_data are imported from eeg_import.py package
    #output: shape(X)=[sample#,channel#, windowSize], sample# depends on the window size and step size (defined inside the funciton get_data()), e.g., we have 100 time points, windowSize=10, stepSize=5, we will have 19 samples
    #shape(y)=[sample#]

    """
    X_train, y_train, X_test, y_test = prepare_data(X, y)
    #shape(X) = [sample, windowSize, matrixHeight, matrixWidth], label y, shape(y) = [sample# ,classes#]

    #don't need to shuffle and split the data X into train and test, we need to predict whole dataset for a new subject (testing subject)
    #only return X_preprocessed, y_oneHot
    return X_train, y_train
    """

    X_preprocessed = prepare_data_test(X)
    # shape(X) = [sample, windowSize, matrixHeight, matrixWidth], label y, shape(y) = [sample# ,classes#]
    # don't need to shuffle and split the data X into train and test, we need to predict whole dataset for a new subject (testing subject)
    # only return X_preprocessed, y_oneHot
    return X_preprocessed, y

#%% (3)Load in model
def loadModel(path = modelPath + "/CNN_full.h5"):
    model = load_model(path)
    return model

#%% (5) Predict - realtime

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

#input  shape(X_preprocessed) = [sample, windowSize, matrixHeight, matrixWidth, 1], label y, shape(y) = [sample  # ,classes#]
def printPred(model):

    count = 0 #sample index, start from 0
    with open('myPred.txt', 'a') as file:
        file.write("subject {}".format(FNAMES[subject_ID]))
    
    while True:
        print("testing sample: {}s - {}s\n".format(count*interval, (count+1)*interval))  # for python 3.5

        X, y = loadData(subject_ID, c=count, interval=interval)
        pred = model.predict(X).argmax(axis=1).tolist()
        y = y.tolist()
        
        # write the reuslt        
        with open('myPred.txt', 'a') as file:
            file.write("\n\ntesting sample: {}s - {}s".format(count*interval, (count+1)*interval))
            file.write("\npredicted label: " + str(max(set(pred), key=pred.count)))
            for uni in set(pred):
                file.write(" " + str(uni) +"-{:.0%}".format(sum(x == uni for x in pred)/len(pred)))
            file.write("\nreal label: {}".format(str(max(set(y), key=y.count))))
            for uni in set(y):
                file.write(" " + str(uni) +"-{:.0%}".format(sum(x == uni for x in y)/len(y)) )

            
        #print one line: Current sample: 0, real label: 2, predicted label: 2, time: 0.0777s
        #see example in expLog.txt
        
        count += 1        
        
        # Pose the process for interval seconds
        print("predicted label: " + str(max(set(pred), key=pred.count)), end="")
        for uni in set(pred):
                print(" " + str(uni) +"-{:.0%}".format(sum(x == uni for x in pred)/len(pred)), end="")
        print("\nreal label: {}".format(str(max(set(y), key=y.count))), end="")
        for uni in set(y):
                print(" " + str(uni) +"-{:.0%}".format(sum(x == uni for x in y)/len(y)), end="")
        print("\nSystem stops for {}s\n\n".format(interval), end="")
        time.sleep(interval)

#%%
if __name__ == "__main__":
    # arguments for this function
    subject_ID = int(sys.argv[1]) # import which subject`s data?
    interval = int(sys.argv[2])  # seconds, e.g., every 5 seconds, classify the data and save the result to the .txt file, given sampling rate=160, we have 800 time points data
    total_running = int(sys.argv[3])  # seconds
    
    print("Test mode started\nsubject: {0}\ninterval: {1}\ntotal running time: {2}"\
                 .format(FNAMES[subject_ID], interval, total_running))
    
    #load existing trained model
    model = loadModel()

    #start timer
    #will stop after running  total_running seconds
    mt = MyTimer().singleShot(total_running, sys.exit)
    
    # Start printing
    printPred(model)

	

	



    

