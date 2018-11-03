import numpy as np
from sklearn.preprocessing import scale

#%%
def convert_mesh(X):
    
    mesh = np.zeros((X.shape[0], X.shape[2], 10, 11, 1))
    X = np.swapaxes(X, 1, 2)
    
    # 1st line
    mesh[:, :, 0, 4:7, 0] = X[:,:,21:24]
    
    # 2nd line
    mesh[:, :, 1, 3:8, 0] = X[:,:,24:29]
    
    # 3rd line
    mesh[:, :, 2, 1:10, 0] = X[:,:,29:38]
    
    # 4th line
    mesh[:, :, 3, 1:10, 0] = np.concatenate((X[:,:,38].reshape(-1, X.shape[1], 1),\
                                          X[:,:,0:7], X[:,:,39].reshape(-1, X.shape[1], 1)), axis=2)
    
    # 5th line
    mesh[:, :, 4, 0:11, 0] = np.concatenate((X[:,:,(42, 40)],\
                                        X[:,:,7:14], X[:,:,(41, 43)]), axis=2)
    
    # 6th line
    mesh[:, :, 5, 1:10, 0] = np.concatenate((X[:,:,44].reshape(-1, X.shape[1], 1),\
                                        X[:,:,14:21], X[:,:,45].reshape(-1, X.shape[1], 1)), axis=2)

               
    # 7th line
    mesh[:, :, 6, 1:10, 0] = X[:,:,46:55] 
    
    # 8th line
    mesh[:, :, 7, 3:8, 0] = X[:,:,55:60] 
    
    # 9th line
    mesh[:, :, 8, 4:7, 0] = X[:,:,60:63] 
    
    # 10th line
    mesh[:, :, 9, 5, 0] = X[:,:,63] 
    
    return mesh

#input:...JImported data by eeg_import.py - dim (Trial*Channel*windowSize)
#output:..JSamples as 2D meshes - dim (Trial*windowSize*height*width*1) -> the last 1 is purely to apply CNN
    
#%%
#input X, shape(X)=[sample#,channel#, windowSize], sample# depends on the window size and step size, e.g., we have 100 time points, windowSize=10, stepSize=5, we will have 19 samples
#step1: one hot encoding for lables
#step2: shuffle data X
#step3: standarize X, Z score
#step4: convert X to 2D matrix, i.e., 10x11 size matrix
#return: shape(X)= [sample,windowSize,matrixHeight,matrixWidth], label y, shape(y)=[sample#,classes#]
def prepare_data_forTesting(X, return_mesh = True):
                                
    # Z-score Normalization
    def scale_data(X):
        shape = X.shape
        for i in range(shape[0]):
            X[i,:, :] = scale(X[i,:, :])
        return X
    
    if return_mesh:
        X_preprocessed  = convert_mesh(scale_data(X))
    
    return X_preprocessed
    
    