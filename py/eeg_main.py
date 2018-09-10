# Modeling & Preprocessing
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, LSTM, Input, TimeDistributed
from keras import initializers, Model, optimizers, callbacks
from keras.utils.training_utils import multi_gpu_model

# Essential Data Handling
import numpy as np

# Modules
from eeg_import import get_data, FNAMES
from eeg_preprocessing import prepare_data

#%%
X,y = get_data(FNAMES[:2], epoch_sec=0.0625)

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

#%%
# Load a model to transfer pre-trained parameters
trans_model = model.load('CNN_3blocks.h5')

# Transfer learning - parameter copy & paste
which_layer = 'CNN1,CNN2,CNN3,batch1,batch2,batch3'.split(',')
layer_names = [layer.name for layer in model.layers]
trans_layer_names = [layer.name for layer in trans_model.layers]

for layer in which_layer:
    ind = layer_names.index(layer)
    trans_ind = trans_layer_names.index(layer)
    model.layers[ind].set_weights(trans_model.layers[trans_ind].get_weights())
    
for layer in model.layers[:9]: # Freeze the first 9 layers(CNN block)
    layer.trainable = False
    
# Turn on multi-GPU mode
model = multi_gpu_model(model, gpus=4)

callbacks_list = [callbacks.ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss'),
                 callbacks.EarlyStopping(monitor='acc', patience=3),
                 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
                 callbacks.TensorBoard(log_dir='./my_log_dir/', histogram=1)]

# Start training
model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['acc'])
model.fit(X_train, y_train, batch_size=64, epochs=5000, validation_data=(X_test, y_test))
