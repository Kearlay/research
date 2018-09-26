'''
Name: eeg_eval.py
Author: Jim Chung
Description:
    This script is written to monitor the performance of neural network trained
    on PhysioNet EEG data. Please check out eeg_main.py or eeg_import_py for
    further information.
    
    'hitory.pkl' file requied in './history/' folder.
'''

# load in libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras import models
from eeg_import import get_data, FNAMES
from eeg_preprocessing import prepare_data
import os

# make directories
if not os.path.exists('./metrics/'):
    os.makedirs('./metrics/')

# functions defined


def plot_history(history):
    loss_list = [s for s in history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history[l], 'b', label='Training loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(str(format(history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./metrics/loss.png")
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(format(history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(format(history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig("./metrics/acc.png")
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.figure(3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./metrics/confuMat.png")
    plt.show()
    
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes):
    
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict(x).argmax(axis=1)
    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true,y_pred)))
    
    print("")
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true,y_pred,digits=4))    
    
    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true,y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix,classes=classes)    
    

# Load in the data
howManyTest = 10
this = np.random.randint(1, 100, size=howManyTest)
X,y = get_data([FNAMES[i] for i in this], epoch_sec=0.0625)
X_train, y_train, X_test, y_test = prepare_data(X, y)

print(X.shape)
print(y.shape)

# Get the model
model = models.load_model('./model/model0.h5')

# Get the history
with open('./history/history0.pkl', 'rb') as hist:
    history = pickle.load(hist)

# Get the graphics
plot_history(history)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
full_multiclass_report(model,
                       X_test,
                       y_test.argmax(axis=1),
                       [1,2,3,4,5])
