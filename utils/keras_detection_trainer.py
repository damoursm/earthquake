#Highly inspired from the following but throughly adapted to our use case:
# - https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/tester.py
# - https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/trainer.py


from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import time

import multiprocessing
from nn.eq_transformer import _lr_schedule
import datetime
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from sklearn.metrics import confusion_matrix, classification_report

import csv
import pandas as pd


def tester(test_dataset,
           model,
           output_name=None,
           batch_size=32):
    
    args = {
    "test_dataset": test_dataset,
    "model": model,
    "output_name": output_name,
    "batch_size": batch_size
    } 

    start_training = time.time()  

    csvTst = open(os.path.join(args['output_name'],'X_test_results.csv'), 'w')          
    test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    test_writer.writerow(['trace_name', 'number_of_detections', 'is_earthquake'])  
    csvTst.flush()

    test_generator = DetectionDataGeneratorPrediction(test_dataset, batch_size=batch_size)

    predD = model.predict_generator(generator=test_generator)

    for ts in range(len(predD)):
        trace_name, p_sample, s_sample, data, is_earthquake = test_dataset.getinfo(ts)
        test_writer.writerow([trace_name, ((predD[ts][0] >= 0.5) * 1), is_earthquake]) 
        csvTst.flush()

    result = pd.read_csv(os.path.join(args['output_name'],'X_test_results.csv'))
    y_true = np.array(result['is_earthquake'] * 1)
    y_pred = np.array(result['number_of_detections'] * 1)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.tight_layout()

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

    fig.savefig(os.path.join(args['output_name'],'X_test_confusion_matrix.png'))

    # Print classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    report_pd = pd.DataFrame(class_report).transpose()
    report_pd.to_csv(os.path.join(args['output_name'],"X_test_classification_report.csv"), index=True)

    end_training = time.time()  
    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta     
                    
    with open(os.path.join(args["output_name"],'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')
        the_file.write('================== Testing Parameters ======================='+'\n')  
        the_file.write('finished the test in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds, 2)))
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('total number of tests '+str(len(args['test_dataset']))+'\n')            
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('confusion matrix: '+str(os.path.join(args['output_name'],"X_test_classification_report.csv"))+'\n')
        the_file.write('classification report: '+str(os.path.join(args['output_name'],"X_test_classification_report.csv"))+'\n')
 


def trainer(train_dataset,
            val_dataset,
            model,
            output_name=None,
            shuffle=True,              
            batch_size=200,
            epochs=200, 
            monitor='val_loss',
            patience=12,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=True,
            best_model_name='best_model.h5'):
        
    """
    
    Train the model.  
    
    Parameters
    ----------
    train_dataset: InstanceDataset
        Dataset for training

    val_dataset: InstanceDataset
        Dataset for validation

    model: model to train
        EqTransformer model

    output_name: str, default=None
        Output directory.

    shuffle: bool, default=True
        To shuffle the list prior to the training.
       
    batch_size: int, default=200
        Batch size.
          
    epochs: int, default=200
        The number of epochs.
          
    monitor: int, default='val_loss'
        The measure used for monitoring.
           
    patience: int, default=12
        The number of epochs without any improvement in the monitoring measure to automatically stop the training.          
           
    gpuid: int, default=None
        Id of GPU used for the prediction. If using CPU set to None. 
         
    gpu_limit: float, default=None
        Set the maximum percentage of memory usage for the GPU.
        
    use_multiprocessing: bool, default=True
        If True, multiple CPUs will be used for the preprocessing of data even when GPU is used for the prediction. 

    Returns
    -------- 
    output_name/output_name_.h5: This is where all good models will be saved.  
    
    output_name/final_model.h5: This is the full model for the last epoch.
    
    output_name/model_weights.h5: These are the weights for the last model.
    
    output_name/history.npy: Training history.
    
    output_name/X_report.txt: A summary of the parameters used for prediction and performance.
    
    output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.
    
    output_name/X_learning_curve_loss.png: The learning curve of loss.
        
    """     

    args = {
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "output_name": output_name,
    "model": model,
    "shuffle": shuffle,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience,                    
    "gpuid": gpuid,
    "gpu_limit": gpu_limit,
    "use_multiprocessing": use_multiprocessing,
    "best_model_name": best_model_name
    }
                       
    def train(args):
        """ 
        
        Performs the training.
    
        Parameters
        ----------
        args : dic
            A dictionary object containing all of the input parameters. 

        Returns
        -------
        history: dic
            Training history.  
            
        model: 
            Trained model.
            
        start_training: datetime
            Training start time. 
            
        end_training: datetime
            Training end time. 
            
        save_dir: str
            Path to the output directory. 
            
        save_models: str
            Path to the folder for saveing the models.  
            
        training size: int
            Number of training samples.
            
        validation size: int
            Number of validation samples.  
            
        """    

        save_dir = args["output_name"] 
        save_models = args["output_name"] 
        callbacks=_make_callback(args, save_models, args["best_model_name"])
        
        # if args['gpuid']:           
        #     os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
        #     tf.Session(config=tf.ConfigProto(log_device_placement=True))
        #     config = tf.ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
        #     K.tensorflow_backend.set_session(tf.Session(config=config))
            
        start_training = time.time()

        training_generator = DetectionDataGenerator(dataset=args["train_dataset"], batch_size=args["batch_size"], shuffle=args["shuffle"])
        validation_generator = DetectionDataGenerator(dataset=args["val_dataset"], batch_size=args["batch_size"], shuffle=args["shuffle"])

        model = args["model"]

        print('Started training ...') 
        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      use_multiprocessing=args['use_multiprocessing'],
                                      workers=multiprocessing.cpu_count(),    
                                      callbacks=callbacks,
                                      epochs=args['epochs'])

        end_training = time.time()  
        
        return history, model, start_training, end_training, save_dir, save_models, len(args["train_dataset"]), len(args["val_dataset"])
                  
    history, model, start_training, end_training, save_dir, save_models, training_size, validation_size=train(args)  
    _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args)


def _make_callback(args, save_models, m_name):
    
    """ 
    
    Generate the callback.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters. 
        
    save_models: str
       Path to the output directory for the models. 
              
    Returns
    -------   
    callbacks: obj
        List of callback objects. 
        
        
    """    

    filepath=os.path.join(save_models, m_name)  
    early_stopping_monitor=EarlyStopping(monitor=args['monitor'], 
                                           patience=args['patience']) 
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=args['monitor'], 
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True)  
    lr_scheduler=LearningRateScheduler(_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args['patience']-2,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    return callbacks
 

def _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args): 

    """ 
    
    Write down the training results.

    Parameters
    ----------
    history: dic
        Training history.  
   
    model: 
        Trained model.  

    start_training: datetime
        Training start time. 

    end_training: datetime
        Training end time.    
         
    save_dir: str
        Path to the output directory. 

    save_models: str
        Path to the folder for saveing the models.  
      
    training_size: int
        Number of training samples.    

    validation_size: int
        Number of validation samples. 

    args: dic
        A dictionary containing all of the input parameters. 
              
    Returns
    -------- 
    ./output_name/history.npy: Training history.    

    ./output_name/X_report.txt: A summary of parameters used for the prediction and perfomance.

    ./output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.         

    ./output_name/X_learning_curve_loss.png: The learning curve of loss.  
        
        
    """   
    
    np.save(save_dir+'/history',history)
    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/final_model_weights.h5')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.legend(['loss'], loc='upper right')  
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.grid(which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
       
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['accuracy'])
    ax.legend(['accuracy'], loc='lower right')        
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.grid(which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_accuracy.png'))) 

    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta    
    
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')
        the_file.write('================== Model Parameters ========================='+'\n')         
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('epochs: '+str(args['epochs'])+'\n')             
        the_file.write('total number of training: '+str(training_size)+'\n')
        the_file.write('total number of validation: '+str(validation_size)+'\n')
        the_file.write('monitor: '+str(args['monitor'])+'\n')
        the_file.write('patience: '+str(args['patience'])+'\n') 
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')  
        the_file.write('================== Training Performance ====================='+'\n')  
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))                         
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last accuracy: '+str(history.history['accuracy'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')




class DetectionDataGenerator(keras.utils.Sequence):
    
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
        X, y1 = self.__data_generation(indexes)
        return X, y1

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
        y1 = np.zeros(self.batch_size)

        X[i, :, :] = data
        y1[i] = detection

        # Generate data
        for i in range(1, len(list_IDs_temp)):
            index = list_IDs_temp[i]
            data, detection, p_phase, s_phase, _ = self.dataset[index]
            data = data.T

            X[i, :, :] = data
            y1[i] = detection           
      
        return X, y1.astype('float32')


class DetectionDataGeneratorPrediction(keras.utils.Sequence):
    
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
        return X

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
 
    