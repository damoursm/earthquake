from __future__ import division, print_function
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from keras import backend as K
from keras.layers import add, Activation, LSTM, Conv1D, InputSpec
from keras.layers import MaxPooling1D, UpSampling1D, Cropping1D, SpatialDropout1D, Bidirectional, BatchNormalization 
from keras.models import Model
from keras.optimizers.legacy import Adam
from obspy.signal.trigger import trigger_onset
import matplotlib
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from utils.dataloaders.InstanceDataset import InstanceDataset


class DataGenerator(keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing 
    
    Parameters
    ----------

    dataset: InstanceDataset
        Dataset to load Instance earthquakes and noise
           
    batch_size: int, default=32
        Batch size.
            
    shuffle: bool, default=True
        Shuffeling the list.

    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 dataset,
                 batch_size=32,
                 shuffle=True):
       
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y1, y2, y3 = self.__data_generation(indexes)
        return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) 
    
    def _scale_amplitude(self, data, rate):
        'Scale amplitude or waveforms'
        
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _drop_channel(self, data, snr, rate):
        'Randomly replace values of one or two components to zeros in earthquake data'

        data = np.copy(data)
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _drop_channel_noise(self, data, rate):
        'Randomly replace values of one or two components to zeros in noise data'
        
        data = np.copy(data)
        if np.random.uniform(0, 1) < rate: 
            c1 = np.random.choice([0, 1])
            c2 = np.random.choice([0, 1])
            c3 = np.random.choice([0, 1])
            if c1 + c2 + c3 > 0:
                data[..., np.array([c1, c2, c3]) == 0] = 0
        return data

    def _add_gaps(self, data, rate): 
        'Randomly add gaps (zeros) of different sizes into waveforms'
        
        data = np.copy(data)
        gap_start = np.random.randint(0, 4000)
        gap_end = np.random.randint(gap_start, 5500)
        if np.random.uniform(0, 1) < rate: 
            data[gap_start:gap_end,:] = 0           
        return data  
    
    def _add_noise(self, data, snr, rate):
        'Randomly add Gaussian noie with a random SNR into waveforms'
        
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
            data_noisy = np.empty((data.shape))
            data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
            data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
            data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
        else:
            data_noisy = data
        return data_noisy   
         
    def _adjust_amplitude_for_multichannels(self, data):
        'Adjust the amplitude of multichaneel data'
        
        tmp = np.max(np.abs(data), axis=0, keepdims=True)
        assert(tmp.shape[-1] == data.shape[-1])
        if np.count_nonzero(tmp) > 0:
          data *= data.shape[-1] / np.count_nonzero(tmp)
        return data

    def _label(self, a=0, b=20, c=40):  
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _add_event(self, data, addp, adds, coda_end, snr, rate): 
        'Add a scaled version of the event into the empty part of the trace'
       
        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                scaleAM = 1/np.random.randint(1, 10)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added
                 
        return data, additions    
    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        'Randomly rotate the array to shift the event location'
        
        org_len = len(data)
        data2 = np.copy(data)
        addp2 = adds2 = coda_end2 = None;
        if np.random.uniform(0, 1) < rate:             
            nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
            data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
            data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
            data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
            if addp+nrotate >= 0 and addp+nrotate < org_len:
                addp2 = addp+nrotate;
            else:
                addp2 = None;
            if adds+nrotate >= 0 and adds+nrotate < org_len:               
                adds2 = adds+nrotate;
            else:
                adds2 = None;                   
            if coda_end+nrotate < org_len:                              
                coda_end2 = coda_end+nrotate 
            else:
                coda_end2 = org_len                 
            if addp2 and adds2:
                data = data2;
                addp = addp2;
                adds = adds2;
                coda_end= coda_end2;                                      
        return data, addp, adds, coda_end      
    
    def _pre_emphasis(self, data, pre_emphasis=0.97):
        'apply the pre_emphasis'

        for ch in range(self.n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data
                    
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'  
        i=0
        index = list_IDs_temp[i]       
        data, detection, p_phase, s_phase, _ = self.dataset[index]
        data = data.T

        X = np.zeros((self.batch_size, data.shape[0], data.shape[1]))
        y1 = np.zeros((self.batch_size, len(detection), 1))
        y2 = np.zeros((self.batch_size, len(p_phase), 1))
        y3 = np.zeros((self.batch_size, len(s_phase), 1))

        X[i, :, :] = data
        y1[i, :, :] = detection[..., np.newaxis]
        y2[i, :, :] = p_phase[..., np.newaxis]
        y3[i, :, :] = s_phase[..., np.newaxis]

        # Generate data
        for i in range(1, len(list_IDs_temp)):
            index = list_IDs_temp[i]
            data, detection, p_phase, s_phase, _ = self.dataset[index]
            data = data.T

            X[i, :, :] = data
            y1[i, :, :] = detection[..., np.newaxis]
            y2[i, :, :] = p_phase[..., np.newaxis]
            y1[i, :, :] = s_phase[..., np.newaxis]              
      
        return X, y1.astype('float32'), y2.astype('float32'), y3.astype('float32')


class DataGeneratorPrediction(keras.utils.Sequence):
    
    """ 
    Keras generator with preprocessing. For prediction. 
    
    Parameters
    ----------
    dataset: InstanceDataset
        Dataset to load Instance earthquakes and noise
            
    batch_size: int, default=32
        Batch size.

        
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
   
    
    """   
    
    def __init__(self, 
                 dataset,
                 batch_size=32):
       
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indexes)
        return ({'input': X})

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
 
    def __data_generation(self, list_IDs_temp):
        'read the waveforms'   
        i=0
        index = list_IDs_temp[i]       
        data, _, _, _, _ = self.dataset[index]
        data = data.T

        X = np.zeros((self.batch_size, data.shape[0], data.shape[1]))

        X[i, :, :] = data

        # Generate data
        for i in range(1, len(list_IDs_temp)):
            index = list_IDs_temp[i]
            data, _, _, _, _ = self.dataset[index]
            data = data.T

            X[i, :, :] = data

        return X
    



def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    ---------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
   ----------------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind




def picker(args, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, spt=None, sst=None):

    """ 
    
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
        Detection probabilities. 
        
    yh2 : 1D array
        P arrival probabilities.  
        
    yh3 : 1D array
        S arrival probabilities. 
        
    yh1_std : 1D array
        Detection standard deviations. 
        
    yh2_std : 1D array
        P arrival standard deviations.  
        
    yh3_std : 1D array
        S arrival standard deviations. 
        
    spt : {int, None}, default=None    
        P arrival time in sample.
        
    sst : {int, None}, default=None
        S arrival time in sample. 
        
   
    Returns
    --------    
    matches: dic
        Contains the information for the detected and picked event.            
        
    matches: dic
        {detection statr-time:[ detection end-time, detection probability, detectin uncertainty, P arrival, P probabiliy, P uncertainty, S arrival,  S probability, S uncertainty]}
            
    pick_errors : dic                
        {detection statr-time:[ P_ground_truth - P_pick, S_ground_truth - S_pick]}
        
    yh3: 1D array             
        normalized S_probability                              
                
    """               
    
 #   yh3[yh3>0.04] = ((yh1+yh3)/2)[yh3>0.04] 
 #   yh2[yh2>0.10] = ((yh1+yh2)/2)[yh2>0.10] 
             
    detection = trigger_onset(yh1, args['detection_threshold'], args['detection_threshold'])
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
          
    P_PICKS = {}
    S_PICKS = {}
    EVENTS = {}
    matches = {}
    pick_errors = {}
    if len(pp_arr) > 0:
        P_uncertainty = None  
            
        for pick in range(len(pp_arr)): 
            pauto = pp_arr[pick]
                        
            if args['estimate_uncertainty'] and pauto:
                P_uncertainty = np.round(yh2_std[int(pauto)], 3)
                    
            if pauto: 
                P_prob = np.round(yh2[int(pauto)], 3) 
                P_PICKS.update({pauto : [P_prob, P_uncertainty]})                 
                
    if len(ss_arr) > 0:
        S_uncertainty = None  
            
        for pick in range(len(ss_arr)):        
            sauto = ss_arr[pick]
                   
            if args['estimate_uncertainty'] and sauto:
                S_uncertainty = np.round(yh3_std[int(sauto)], 3)
                    
            if sauto: 
                S_prob = np.round(yh3[int(sauto)], 3) 
                S_PICKS.update({sauto : [S_prob, S_uncertainty]})             
            
    if len(detection) > 0:
        D_uncertainty = None  
        
        for ev in range(len(detection)):                                 
            if args['estimate_uncertainty']:               
                D_uncertainty = np.mean(yh1_std[detection[ev][0]:detection[ev][1]])
                D_uncertainty = np.round(D_uncertainty, 3)
                    
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)
                    
            EVENTS.update({ detection[ev][0] : [D_prob, D_uncertainty, detection[ev][1]]})            
    
    # matching the detection and picks
    def pair_PS(l1, l2, dist):
        l1.sort()
        l2.sort()
        b = 0
        e = 0
        ans = []
        
        for a in l1:
            while l2[b] and b < len(l2) and a - l2[b] > dist:
                b += 1
            while l2[e] and e < len(l2) and l2[e] - a <= dist:
                e += 1
            ans.extend([[a,x] for x in l2[b:e]])
            
        best_pair = None
        for pr in ans: 
            ds = pr[1]-pr[0]
            if abs(ds) < dist:
                best_pair = pr
                dist = ds           
        return best_pair


    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][2]
        S_error = None
        P_error = None        
        if int(ed-bg) >= 10:
                                    
            candidate_Ss = {}
            for Ss, S_val in S_PICKS.items():
                if Ss > bg and Ss < ed:
                    candidate_Ss.update({Ss : S_val}) 
             
            if len(candidate_Ss) > 1:                
# =============================================================================
#                 Sr_st = 0
#                 buffer = {}
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     if S_valCan[0] > Sr_st:
#                         buffer = {SsCan : S_valCan}
#                         Sr_st = S_valCan[0]
#                 candidate_Ss = buffer
# =============================================================================              
                candidate_Ss = {list(candidate_Ss.keys())[0] : candidate_Ss[list(candidate_Ss.keys())[0]]}


            if len(candidate_Ss) == 0:
                    candidate_Ss = {None:[None, None]}

            candidate_Ps = {}
            for Ps, P_val in P_PICKS.items():
                if list(candidate_Ss)[0]:
                    if Ps > bg-100 and Ps < list(candidate_Ss)[0]-10:
                        candidate_Ps.update({Ps : P_val}) 
                else:         
                    if Ps > bg-100 and Ps < ed:
                        candidate_Ps.update({Ps : P_val}) 
                    
            if len(candidate_Ps) > 1:
                Pr_st = 0
                buffer = {}
                for PsCan, P_valCan in candidate_Ps.items():
                    if P_valCan[0] > Pr_st:
                        buffer = {PsCan : P_valCan} 
                        Pr_st = P_valCan[0]
                candidate_Ps = buffer
                    
            if len(candidate_Ps) == 0:
                    candidate_Ps = {None:[None, None]}
                    
                    
# =============================================================================
#             Ses =[]; Pes=[]
#             if len(candidate_Ss) >= 1:
#                 for SsCan, S_valCan in candidate_Ss.items():
#                     Ses.append(SsCan) 
#                                 
#             if len(candidate_Ps) >= 1:
#                 for PsCan, P_valCan in candidate_Ps.items():
#                     Pes.append(PsCan) 
#             
#             if len(Ses) >=1 and len(Pes) >= 1:
#                 PS = pair_PS(Pes, Ses, ed-bg)
#                 if PS:
#                     candidate_Ps = {PS[0] : candidate_Ps.get(PS[0])}
#                     candidate_Ss = {PS[1] : candidate_Ss.get(PS[1])}
# =============================================================================

            if list(candidate_Ss)[0] or list(candidate_Ps)[0]:                 
                matches.update({
                                bg:[ed, 
                                    EVENTS[ev][0], 
                                    EVENTS[ev][1], 
                                
                                    list(candidate_Ps)[0],  
                                    candidate_Ps[list(candidate_Ps)[0]][0], 
                                    candidate_Ps[list(candidate_Ps)[0]][1],  
                                                
                                    list(candidate_Ss)[0],  
                                    candidate_Ss[list(candidate_Ss)[0]][0], 
                                    candidate_Ss[list(candidate_Ss)[0]][1],  
                                                ] })
                
                if sst and sst > bg and sst < EVENTS[ev][2]:
                    if list(candidate_Ss)[0]:
                        S_error = sst -list(candidate_Ss)[0] 
                    else:
                        S_error = None
                                            
                if spt and spt > bg-100 and spt < EVENTS[ev][2]:
                    if list(candidate_Ps)[0]:  
                        P_error = spt - list(candidate_Ps)[0] 
                    else:
                        P_error = None
                                          
                pick_errors.update({bg:[P_error, S_error]})
      
    return matches, pick_errors, yh3





def generate_arrays_from_file(file_list, step):
    
    """ 
    
    Make a generator to generate list of trace names.
    
    Parameters
    ----------
    file_list : str
        A list of trace names.  
        
    step : int
        Batch size.  
        
    Returns
    --------  
    chunck : str
        A batch of trace names. 
        
    """     
    
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck   

    
    

def f1(y_true, y_pred):
    
    """ 
    
    Calculate F1-score.
    
    Parameters
    ----------
    y_true : 1D array
        Ground truth labels. 
        
    y_pred : 1D array
        Predicted labels.     
        
    Returns
    -------  
    f1 : float
        Calculated F1-score. 
        
    """     
    
    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def normalize(data, mode='std'):
    
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """   
    
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              
    elif mode == 'std':        
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data
    
    

def _block_BiLSTM(filters, drop_rate, padding, inpR):
    'Returns LSTM residual block'    
    prev = inpR
    x_rnn = Bidirectional(LSTM(filters, return_sequences=True, dropout=drop_rate, recurrent_dropout=drop_rate))(prev)
    NiN = Conv1D(filters, 1, padding = padding)(x_rnn)     
    res_out = BatchNormalization()(NiN)
    return res_out


def _block_CNN_1(filters, ker, drop_rate, activation, padding, inpC): 
    ' Returns CNN residual blocks '
    prev = inpC
    layer_1 = BatchNormalization()(prev) 
    act_1 = Activation(activation)(layer_1) 
    act_1 = SpatialDropout1D(drop_rate)(act_1, training=True)
    conv_1 = Conv1D(filters, ker, padding = padding)(act_1) 
    
    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation(activation)(layer_2) 
    act_2 = SpatialDropout1D(drop_rate)(act_2, training=True)
    conv_2 = Conv1D(filters, ker, padding = padding)(act_2)
    
    res_out = add([prev, conv_2])
    
    return res_out 


def _encoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the encoder that is a combination of residual blocks and maxpooling.'        
    e = inpC
    for dp in range(depth):
        e = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer=keras.regularizers.l2(ker_regul),
                   bias_regularizer=keras.regularizers.l1(bias_regul),
                   )(e)             
        e = MaxPooling1D(2, padding = padding)(e)            
    return(e) 


def _decoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the dencoder that is a combination of residual blocks and upsampling. '           
    d = inpC
    for dp in range(depth):        
        d = UpSampling1D(2)(d) 
        if dp == 2:
            d = Cropping1D(cropping=(1, 1))(d)           
        d = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer=keras.regularizers.l2(ker_regul),
                   bias_regularizer=keras.regularizers.l1(bias_regul),
                   )(d)        
    return(d)  
 


def _lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class EncodeDecodeModel():
    """ 
    
    Creates the model
    
    Parameters
    ----------
    nb_filters: list
        The list of filter numbers. 
        
    kernel_size: list
        The size of the kernel to use in each convolutional layer.
        
    padding: str
        The padding to use in the convolutional layers.

    activationf: str
        Activation funciton type.

    endcoder_depth: int
        The number of layers in the encoder.
        
    decoder_depth: int
        The number of layers in the decoder.

    cnn_blocks: int
        The number of residual CNN blocks.

    BiLSTM_blocks: int=
        The number of Bidirectional LSTM blocks.
  
    drop_rate: float 
        Dropout rate.

    loss_weights: list
        Weights of the loss function for the detection, P picking, and S picking.       
                
    loss_types: list
        Types of the loss function for the detection, P picking, and S picking. 

    kernel_regularizer: str
        l1 norm regularizer.

    bias_regularizer: str
        l1 norm regularizer.
           
    Returns
    ----------
        The complied model: keras model
        
    """

    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 BiLSTM_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                 kernel_regularizer = 1e-4,
                 bias_regularizer = 1e-3
                 ):
        
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth= endcoder_depth
        self.decoder_depth= decoder_depth
        self.cnn_blocks= cnn_blocks
        self.BiLSTM_blocks= BiLSTM_blocks     
        self.drop_rate= drop_rate
        self.loss_weights= loss_weights  
        self.loss_types = loss_types       
        self.kernel_regularizer = kernel_regularizer     
        self.bias_regularizer = bias_regularizer 

        
    def __call__(self, inp):

        x = inp
        x = _encoder(self.nb_filters, 
                    self.kernel_size, 
                    self.endcoder_depth, 
                    self.drop_rate, 
                    self.kernel_regularizer, 
                    self.bias_regularizer,
                    self.activationf, 
                    self.padding,
                    x)    
        
        last_filter = self.nb_filters[self.endcoder_depth - 1]
        for cb in range(self.cnn_blocks):
            x = _block_CNN_1(last_filter, 3, self.drop_rate, self.activationf, self.padding, x)

        for bb in range(self.BiLSTM_blocks):
            x = _block_BiLSTM(self.nb_filters[1], self.drop_rate, self.padding, x)

        encoded = x            
            
        decoder_D = _decoder([i for i in reversed(self.nb_filters)], 
                             [i for i in reversed(self.kernel_size)], 
                             self.decoder_depth, 
                             self.drop_rate, 
                             self.kernel_regularizer, 
                             self.bias_regularizer,
                             self.activationf, 
                             self.padding,                             
                             encoded)
        d = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='detector')(decoder_D)


        PLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        
        decoder_P = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)], 
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            PLSTM)
        P = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_P')(decoder_P)
        

        SLSTM = LSTM(self.nb_filters[1], return_sequences=True, dropout=self.drop_rate, recurrent_dropout=self.drop_rate)(encoded)
        decoder_S = _decoder([i for i in reversed(self.nb_filters)], 
                            [i for i in reversed(self.kernel_size)],
                            self.decoder_depth, 
                            self.drop_rate, 
                            self.kernel_regularizer, 
                            self.bias_regularizer,
                            self.activationf, 
                            self.padding,                            
                            SLSTM) 
        S = Conv1D(1, 11, padding = self.padding, activation='sigmoid', name='picker_S')(decoder_S)
        

        model = Model(inputs=inp, outputs=[d, P, S])

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,    
            optimizer=Adam(learning_rate=_lr_schedule(0)), metrics=[f1])

        return model

        

