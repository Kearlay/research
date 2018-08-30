def get_data(subj_num, epoch_sec=0.0625):
    """ Import each subject`s trials and make a 3D array
        Output shape: (Trial*Channel*TimeFrames)
        
        Some edf+ files recorded at low sampling rate, 128Hz, are excluded. 
        Majority was sampled at 160Hz.
        
        epoch_sec: time interval for one segment of mashes
        """
    
    # To calculated completion rate
    count = 0
    
    # Initiate X, y
    X = []
    y = []
    
    # fixed numbers
    nChan = 64 
    sfreq = 160
    sliding = epoch_sec/2 
    
    # Sub-function to assign X and X, y
    def append_X(n_segments, old_x):
        new_x = old_x + [data[:, int(sfreq*sliding*n):int(sfreq*sliding*(n+2))] for n in range(n_segments)\
                     if data[:, int(sfreq*sliding*n):int(sfreq*sliding*(n+2))].shape==(nChan, int(sfreq*epoch_sec))]
        return new_x
    
    def append_X_Y(run_type, event, old_x, old_y):
        # Number of sliding windows
        n_segments = int(event[1]/epoch_sec)*2-1
        
        # Instantiate new_x, new_y
        new_y = old_y
        new_x = old_x
        
        # y assignment
        if run_type == 1:
            if event[2] == 'T1':
                new_y = old_y + [1]*n_segments
                new_x = append_X(n_segments, old_x)

            elif event[2] == 'T2':
                new_y = old_y + [2]*n_segments
                new_x = append_X(n_segments, old_x)
        
        if run_type == 2:
            if event[2] == 'T1':
                new_y = old_y + [3]*n_segments
                new_x = append_X(n_segments, old_x)
            
            elif event[2] == 'T2':
                new_y = old_y + [4]*n_segments
                new_x = append_X(n_segments, old_x)
        
        return new_x, new_y
    
    # Iterate over subj_num: S001, S002, S003...
    for subj in subj_num:
        # Return completion rate
        count+=1
        print('working on {}, {:.1%} completed'.format(subj, count/len(subj_num)))

        # Get file names
        fnames = glob(os.path.join(PATH, subj, subj+'R*.edf'))
        fnames = [name for name in fnames if name[-6:-4] in run_type_0+run_type_1+run_type_2]
        
        for i, fname in enumerate(fnames):
            
            # Import data into MNE raw object
            raw = read_raw_edf(fname, preload=True, verbose=False)
            picks = pick_types(raw.info, eeg=True)
            
            if raw.info['sfreq'] != 160:
                print(f'{subj} is sampled at 128Hz so will be excluded.')
                break
            
            # Get annotation
            events = raw.find_edf_events()
            
            # Get data
            data = raw.get_data(picks=picks)
            
            # Number of this run
            which_run = fname[-6:-4]
            
            """ Assignment Starts """ 
            # run 1 - baseline (eye closed)
            if which_run in run_type_0:

                # Number of sliding windows
                n_segments = int((raw.n_times/(epoch_sec*sfreq))*2-1)
                
                # Append 0`s based on number of windows
                y.extend([0]*n_segments)
                X = append_X(n_segments, X)
                    
            # run 4,8,12 - imagine opening and closing left or right fist    
            elif which_run in run_type_1:
                
                for i, event in enumerate(events):
                    X, y = append_X_Y(run_type=1, event=event, old_x=X, old_y=y)
                        
            # run 6,10,14 - imagine opening and closing both fists or both feet
            elif which_run in run_type_2:
                   
                for i, event in enumerate(events):         
                    X, y = append_X_Y(run_type=2, event=event, old_x=X, old_y=y)
                        
    X = np.stack(X)
    y = np.array(y).reshape((-1,1))
    return X, y