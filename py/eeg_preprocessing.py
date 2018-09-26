import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#%%
def convert_mesh(X):
    
    mesh = np.zeros((X.shape[0], X.shape[2], 10, 11))
    X = np.swapaxes(X, 1, 2)
    
    # 1st line
    mesh[:, :, 0, 4:7] = X[:,:,21:24]; print('1st finished')
    
    # 2nd line
    mesh[:, :, 1, 3:8] = X[:,:,24:29]; print('2nd finished')
    
    # 3rd line
    mesh[:, :, 2, 1:10] = X[:,:,29:38]; print('3rd finished')
    
    # 4th line
    mesh[:, :, 3, 1:10] = np.concatenate((X[:,:,38].reshape(-1, X.shape[1], 1),\
                                          X[:,:,0:7], X[:,:,39].reshape(-1, X.shape[1], 1)), axis=2)
    print('4th finished')
    
    # 5th line
    mesh[:, :, 4, 0:11] = np.concatenate((X[:,:,(42, 40)],\
                                        X[:,:,7:14], X[:,:,(41, 43)]), axis=2)
    print('5th finished')
    
    # 6th line
    mesh[:, :, 5, 1:10] = np.concatenate((X[:,:,44].reshape(-1, X.shape[1], 1),\
                                        X[:,:,14:21], X[:,:,45].reshape(-1, X.shape[1], 1)), axis=2)
    print('6th finished')
               
    # 7th line
    mesh[:, :, 6, 1:10] = X[:,:,46:55]; print('7th finished')
    
    # 8th line
    mesh[:, :, 7, 3:8] = X[:,:,55:60]; print('8th finished')
    
    # 9th line
    mesh[:, :, 8, 4:7] = X[:,:,60:63]; print('9th finished')
    
    # 10th line
    mesh[:, :, 9, 5] = X[:,:,63]; print('10th finished')
    
    return mesh

#%%
def prepare_data(X, y, test_ratio=0.2, return_mesh=True, set_seed=42):
    
    # y encoding
    oh = OneHotEncoder()
    y = oh.fit_transform(y).toarray()
    
    # Shuffle trials
    np.random.seed(set_seed)
    trials = X.shape[0]
    shuffle_indices = np.random.permutation(trials)
    X = X[shuffle_indices]
    y = y[shuffle_indices]
    
    # Test set seperation
    train_size = int(trials*(1-test_ratio)) 
    X_train, X_test, y_train, y_test = X[:train_size,:,:], X[train_size:,:,:],\
                                    y[:train_size,:], y[train_size:,:]
                                    
    # Z-score Normalization
    def scale_data(X):
        shape = X.shape
        scaler = StandardScaler()
        scaled_X = np.zeros((shape[0], shape[1], shape[2]))
        displayStep = max(int(shape[0]/10), 1)
        for i in range(shape[0]):
            for z in range(shape[2]):
                scaled_X[i, :, z] = np.squeeze(scaler.fit_transform(X[i, :, z].reshape(-1, 1)))
            if i%displayStep == 0:
                print('{:.1%} done'.format((i+1)/shape[0]))   
        return scaled_X
            
    X_train, X_test  = scale_data(X_train), scale_data(X_test)
    
    if return_mesh:
        X_train, X_test = convert_mesh(X_train), convert_mesh(X_test)
    
    return X_train, y_train, X_test, y_test
    
    