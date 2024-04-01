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


# class DataGenerator(keras.utils.Sequence):
    
#     """ 
    
#     Keras generator with preprocessing 
    
#     Parameters
#     ----------

#     dataset: InstanceDataset
#         Dataset to load Instance earthquakes and noise
           
#     batch_size: int, default=32
#         Batch size.
            
#     shuffle: bool, default=True
#         Shuffeling the list.

#     Returns
#     --------        
#     Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
#     """   
    
#     def __init__(self, 
#                  dataset,
#                  batch_size=32,
#                  shuffle=True):
       
#         'Initialization'
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.dataset) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         X, y1, y2, y3 = self.__data_generation(indexes)
#         return ({'input': X}, {'detector': y1, 'picker_P': y2, 'picker_S': y3})

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.dataset))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes) 
    
#     def _scale_amplitude(self, data, rate):
#         'Scale amplitude or waveforms'
        
#         tmp = np.random.uniform(0, 1)
#         if tmp < rate:
#             data *= np.random.uniform(1, 3)
#         elif tmp < 2*rate:
#             data /= np.random.uniform(1, 3)
#         return data

#     def _drop_channel(self, data, snr, rate):
#         'Randomly replace values of one or two components to zeros in earthquake data'

#         data = np.copy(data)
#         if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
#             c1 = np.random.choice([0, 1])
#             c2 = np.random.choice([0, 1])
#             c3 = np.random.choice([0, 1])
#             if c1 + c2 + c3 > 0:
#                 data[..., np.array([c1, c2, c3]) == 0] = 0
#         return data

#     def _drop_channel_noise(self, data, rate):
#         'Randomly replace values of one or two components to zeros in noise data'
        
#         data = np.copy(data)
#         if np.random.uniform(0, 1) < rate: 
#             c1 = np.random.choice([0, 1])
#             c2 = np.random.choice([0, 1])
#             c3 = np.random.choice([0, 1])
#             if c1 + c2 + c3 > 0:
#                 data[..., np.array([c1, c2, c3]) == 0] = 0
#         return data

#     def _add_gaps(self, data, rate): 
#         'Randomly add gaps (zeros) of different sizes into waveforms'
        
#         data = np.copy(data)
#         gap_start = np.random.randint(0, 4000)
#         gap_end = np.random.randint(gap_start, 5500)
#         if np.random.uniform(0, 1) < rate: 
#             data[gap_start:gap_end,:] = 0           
#         return data  
    
#     def _add_noise(self, data, snr, rate):
#         'Randomly add Gaussian noie with a random SNR into waveforms'
        
#         data_noisy = np.empty((data.shape))
#         if np.random.uniform(0, 1) < rate and all(snr >= 10.0): 
#             data_noisy = np.empty((data.shape))
#             data_noisy[:, 0] = data[:,0] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,0]), data.shape[0])
#             data_noisy[:, 1] = data[:,1] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,1]), data.shape[0])
#             data_noisy[:, 2] = data[:,2] + np.random.normal(0, np.random.uniform(0.01, 0.15)*max(data[:,2]), data.shape[0])    
#         else:
#             data_noisy = data
#         return data_noisy   
         
#     def _adjust_amplitude_for_multichannels(self, data):
#         'Adjust the amplitude of multichaneel data'
        
#         tmp = np.max(np.abs(data), axis=0, keepdims=True)
#         assert(tmp.shape[-1] == data.shape[-1])
#         if np.count_nonzero(tmp) > 0:
#           data *= data.shape[-1] / np.count_nonzero(tmp)
#         return data

#     def _label(self, a=0, b=20, c=40):  
#         'Used for triangolar labeling'
        
#         z = np.linspace(a, c, num = 2*(b-a)+1)
#         y = np.zeros(z.shape)
#         y[z <= a] = 0
#         y[z >= c] = 0
#         first_half = np.logical_and(a < z, z <= b)
#         y[first_half] = (z[first_half]-a) / (b-a)
#         second_half = np.logical_and(b < z, z < c)
#         y[second_half] = (c-z[second_half]) / (c-b)
#         return y

#     def _add_event(self, data, addp, adds, coda_end, snr, rate): 
#         'Add a scaled version of the event into the empty part of the trace'
       
#         added = np.copy(data)
#         additions = None
#         spt_secondEV = None
#         sst_secondEV = None
#         if addp and adds:
#             s_p = adds - addp
#             if np.random.uniform(0, 1) < rate and all(snr>=10.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
#                 secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
#                 scaleAM = 1/np.random.randint(1, 10)
#                 space = data.shape[0]-secondEV_strt  
#                 added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*scaleAM
#                 added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*scaleAM 
#                 added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*scaleAM          
#                 spt_secondEV = secondEV_strt   
#                 if  spt_secondEV + s_p + 21 <= data.shape[0]:
#                     sst_secondEV = spt_secondEV + s_p
#                 if spt_secondEV and sst_secondEV:                                                                     
#                     additions = [spt_secondEV, sst_secondEV] 
#                     data = added
                 
#         return data, additions    
    
    
#     def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
#         'Randomly rotate the array to shift the event location'
        
#         org_len = len(data)
#         data2 = np.copy(data)
#         addp2 = adds2 = coda_end2 = None;
#         if np.random.uniform(0, 1) < rate:             
#             nrotate = int(np.random.uniform(1, int(org_len - coda_end)))
#             data2[:, 0] = list(data[:, 0])[-nrotate:] + list(data[:, 0])[:-nrotate]
#             data2[:, 1] = list(data[:, 1])[-nrotate:] + list(data[:, 1])[:-nrotate]
#             data2[:, 2] = list(data[:, 2])[-nrotate:] + list(data[:, 2])[:-nrotate]
                    
#             if addp+nrotate >= 0 and addp+nrotate < org_len:
#                 addp2 = addp+nrotate;
#             else:
#                 addp2 = None;
#             if adds+nrotate >= 0 and adds+nrotate < org_len:               
#                 adds2 = adds+nrotate;
#             else:
#                 adds2 = None;                   
#             if coda_end+nrotate < org_len:                              
#                 coda_end2 = coda_end+nrotate 
#             else:
#                 coda_end2 = org_len                 
#             if addp2 and adds2:
#                 data = data2;
#                 addp = addp2;
#                 adds = adds2;
#                 coda_end= coda_end2;                                      
#         return data, addp, adds, coda_end      
    
#     def _pre_emphasis(self, data, pre_emphasis=0.97):
#         'apply the pre_emphasis'

#         for ch in range(self.n_channels): 
#             bpf = data[:, ch]  
#             data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
#         return data
                    
#     def __data_generation(self, list_IDs_temp):
#         'read the waveforms'  
#         i=0
#         index = list_IDs_temp[i]       
#         data, detection, p_phase, s_phase, _ = self.dataset[index]
#         data = data.T

#         X = np.zeros((self.batch_size, data.shape[0], data.shape[1]))
#         y1 = np.zeros((self.batch_size, len(detection), 1))
#         y2 = np.zeros((self.batch_size, len(p_phase), 1))
#         y3 = np.zeros((self.batch_size, len(s_phase), 1))

#         X[i, :, :] = data
#         y1[i, :, :] = detection[..., np.newaxis]
#         y2[i, :, :] = p_phase[..., np.newaxis]
#         y3[i, :, :] = s_phase[..., np.newaxis]

#         # Generate data
#         for i in range(1, len(list_IDs_temp)):
#             index = list_IDs_temp[i]
#             data, detection, p_phase, s_phase, _ = self.dataset[index]
#             data = data.T

#             X[i, :, :] = data
#             y1[i, :, :] = detection[..., np.newaxis]
#             y2[i, :, :] = p_phase[..., np.newaxis]
#             y1[i, :, :] = s_phase[..., np.newaxis]              
      
#         return X, y1.astype('float32'), y2.astype('float32'), y3.astype('float32')


# class DataGeneratorPrediction(keras.utils.Sequence):
    
#     """ 
#     Keras generator with preprocessing. For prediction. 
    
#     Parameters
#     ----------
#     dataset: InstanceDataset
#         Dataset to load Instance earthquakes and noise
            
#     batch_size: int, default=32
#         Batch size.

        
#     Returns
#     --------        
#     Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
   
    
#     """   
    
#     def __init__(self, 
#                  dataset,
#                  batch_size=32):
       
#         'Initialization'
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.dataset) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         X = self.__data_generation(indexes)
#         return ({'input': X})

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.dataset))
 
#     def __data_generation(self, list_IDs_temp):
#         'read the waveforms'   
#         i=0
#         index = list_IDs_temp[i]       
#         data, _, _, _, _ = self.dataset[index]
#         data = data.T

#         X = np.zeros((self.batch_size, data.shape[0], data.shape[1]))

#         X[i, :, :] = data

#         # Generate data
#         for i in range(1, len(list_IDs_temp)):
#             index = list_IDs_temp[i]
#             data, _, _, _, _ = self.dataset[index]
#             data = data.T

#             X[i, :, :] = data

#         return X
    
   

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

        

