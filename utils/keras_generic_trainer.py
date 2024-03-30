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
from nn.eq_transformer import DataGenerator, DataGeneratorPrediction, _lr_schedule
import datetime
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def tester(test_dataset,
           model,
           batch_size=32):

    test_generator = DataGeneratorPrediction(test_dataset, batch_size=batch_size)

    predD, predP, predS = model.predict_generator(generator=test_generator)


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
        
    model: eq transformer model

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
    
    output_name/test.npy: A number list containing the trace names for the test set.
    
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
        callbacks=_make_callback(args, save_models, args["best_model_name.h5"])
        
        if args['gpuid']:           
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
            tf.Session(config=tf.ConfigProto(log_device_placement=True))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
            K.tensorflow_backend.set_session(tf.Session(config=config))
            
        start_training = time.time()

        training_generator = DataGenerator(dataset=args["train_dataset"], batch_size=args["batch_size"], shuffle=args["shuffle"])
        validation_generator = DataGenerator(dataset=args["val_dataset"], batch_size=args["batch_size"], shuffle=args["shuffle"])

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
    ax.plot(history.history['detector_loss'])
    ax.plot(history.history['picker_P_loss'])
    ax.plot(history.history['picker_S_loss'])
    try:
        ax.plot(history.history['val_loss'], '--')
        ax.plot(history.history['val_detector_loss'], '--')
        ax.plot(history.history['val_picker_P_loss'], '--')
        ax.plot(history.history['val_picker_S_loss'], '--') 
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss', 
               'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss'], loc='upper right')
    except Exception:
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss'], loc='upper right')  
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.grid(which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
       
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['detector_f1'])
    ax.plot(history.history['picker_P_f1'])
    ax.plot(history.history['picker_S_f1'])
    try:
        ax.plot(history.history['val_detector_f1'], '--')
        ax.plot(history.history['val_picker_P_f1'], '--')
        ax.plot(history.history['val_picker_S_f1'], '--')
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1', 'val_detector_f1', 'val_picker_P_f1', 'val_picker_S_f1'], loc='lower right')
    except Exception:
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1'], loc='lower right')        
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    # plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.grid(which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_f1.png'))) 

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
        # the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
        # the_file.write('input_csv: '+str(args['input_csv'])+'\n')
        # the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Model Parameters ========================='+'\n')   
        # # the_file.write('input_dimention: '+str(args['input_dimention'])+'\n')
        # the_file.write('cnn_blocks: '+str(args['cnn_blocks'])+'\n')
        # the_file.write('lstm_blocks: '+str(args['lstm_blocks'])+'\n')
        # the_file.write('padding_type: '+str(args['padding'])+'\n')
        # the_file.write('activation_type: '+str(args['activation'])+'\n')        
        # the_file.write('drop_rate: '+str(args['drop_rate'])+'\n')            
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')  
        # the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        # the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
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
        the_file.write('last detector_loss: '+str(history.history['detector_loss'][-1])+'\n')
        the_file.write('last picker_P_loss: '+str(history.history['picker_P_loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['picker_S_loss'][-1])+'\n')
        the_file.write('last detector_f1: '+str(history.history['detector_f1'][-1])+'\n')
        the_file.write('last picker_P_f1: '+str(history.history['picker_P_f1'][-1])+'\n')
        the_file.write('last picker_S_f1: '+str(history.history['picker_S_f1'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')               
        # the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')

