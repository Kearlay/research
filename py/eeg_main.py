# Modeling & Preprocessing
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, LSTM, Input, TimeDistributed
from keras import initializers, Model, optimizers, callbacks
from keras.models import load_model
#from keras.utils.training_utils import multi_gpu_model

from glob import glob

# Modules
from eeg_import import get_data, FNAMES
from eeg_preprocessing import prepare_data

# Save the model
import pickle

# Make directories for model binaries
import os
DIR = ['./model', './history']
for directory in DIR:
    if not os.path.exists(directory):
        os.makedirs(directory)


#%%
X,y = get_data(FNAMES, epoch_sec=0.0625)

print(X.shape)
print(y.shape)

#%%

X_train, y_train, X_test, y_test = prepare_data(X, y)

del X
del y
#%%

# Make another dimension, 1, to apply CNN for each time frame.
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

## Complicated Model - the same as Zhang`s
input_shape = (10, 10, 11, 1)
lecun = initializers.lecun_normal(seed=42)

# TimeDistributed Wrapper
def timeDist(layer, prev_layer, name):
    return TimeDistributed(layer, name=name)(prev_layer)
    

# Input layer
inputs = Input(shape=input_shape)

# Convolutional layers block
x = timeDist(Conv2D(32, (3,3), padding='same', 
                    data_format='channels_last', kernel_initializer=lecun), inputs, name='CNN1')
x = BatchNormalization(name='batch1')(x)
x = Activation('elu', name='act1')(x)
x = timeDist(Conv2D(64, (3,3), padding='same', data_format='channels_last', kernel_initializer=lecun), x, name='CNN2')
x = BatchNormalization(name='batch2')(x)
x = Activation('elu', name='act2')(x)
x = timeDist(Conv2D(128, (3,3), padding='same', data_format='channels_last', kernel_initializer=lecun), x, name='CNN3')
x = BatchNormalization(name='batch3')(x)
x = Activation('elu', name='act3')(x)
x = timeDist(Flatten(), x, name='flatten')

# Fully connected layer block
y = Dense(1024, kernel_initializer=lecun, name='FC')(x)
y = Dropout(0.5, name='dropout1')(y)
y = BatchNormalization(name='batch4')(y)
y = Activation(activation='elu')(y)

# Recurrent layers block
z = LSTM(64, kernel_initializer=lecun, return_sequences=True, name='LSTM1')(y)
z = LSTM(64, kernel_initializer=lecun, name='LSTM2')(z)

# Fully connected layer block
h = Dense(1024, kernel_initializer=lecun, activation='elu', name='FC2')(z)
h = Dropout(0.5, name='dropout2')(h)

# Output layer
outputs = Dense(5, activation='softmax')(h)

# Model compile
model = Model(inputs=inputs, outputs=outputs)
model.summary()

# Get past models

MODEL_LIST = []#glob('./model/*')
'''
if MODEL_LIST:
    print('A model that already exists detected and loaded.')
    model = load_model(MODEL_LIST[-1])
'''
    
callbacks_list = [callbacks.ModelCheckpoint('./model/model' + str(len(MODEL_LIST)) + '.h5', 
                                            save_best_only=True, 
                                            monitor='val_loss'),
                 callbacks.EarlyStopping(monitor='acc', patience=10),
                 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
                 callbacks.TensorBoard(log_dir='./my_log_dir', 
                                       histogram_freq=0, 
                                       write_graph=True,
                                       write_images=True)]

# Start training
model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['acc'])
hist = model.fit(X_train, y_train, batch_size=64, epochs=3, 
                callbacks=callbacks_list, validation_data=(X_test, y_test))

# Save the history
hist_list = []#glob('./history/*')
count = len(hist_list)
FILE_NAME = './history/history' + str(count) +'.pkl'

with open(FILE_NAME, 'wb') as object:
    pickle.dump(hist.history, object)
    
# This is how to load the history
'''
with open('./history/history2.pkl', 'rb') as ob:
    data = pickle.load(ob)
print(data)
'''
