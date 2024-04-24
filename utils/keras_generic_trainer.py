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
from nn.eq_transformer import DataGenerator, DataGeneratorPrediction, _lr_schedule
import datetime
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from sklearn.metrics import confusion_matrix, classification_report

import csv
import pandas as pd

from utils.picker_utils import picker


def tester(test_dataset,
           model,
           output_name=None,
           detection_threshold=0.50,
           P_threshold=0.30, 
           S_threshold=0.30,
           estimate_uncertainty=True, 
           number_of_sampling=5,
           number_of_plots=100, 
           loss_weights=[0.05, 0.40, 0.55],
           loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
           batch_size=32):
    
    args = {
    "test_dataset": test_dataset,
    "model": model,
    "output_name": output_name,
    "detection_threshold": detection_threshold,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "estimate_uncertainty": estimate_uncertainty,
    "number_of_sampling": number_of_sampling,
    "number_of_plots": number_of_plots,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "batch_size": batch_size
    } 

    start_training = time.time()  

    csvTst = open(os.path.join(args['output_name'],'X_test_results.csv'), 'w')          
    test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    test_writer.writerow(['trace_name', 
                          'p_arrival_sample',
                          's_arrival_sample', 
                          
                          'number_of_detections',
                          'detection_probability',
                          'detection_uncertainty',
                          
                          'P_pick', 
                          'P_probability',
                          'P_uncertainty',
                          'P_error',
                          
                          'S_pick',
                          'S_probability',
                          'S_uncertainty', 
                          'S_error',

                          'is_earthquake'
                          ])  
    csvTst.flush()

    test_generator = DataGeneratorPrediction(test_dataset, batch_size=batch_size)

    if args['estimate_uncertainty']:
        pred_DD = []
        pred_PP = []
        pred_SS = []          
        for mc in range(args['number_of_sampling']):
            predD, predP, predS = model.predict_generator(generator=test_generator)
            pred_DD.append(predD)
            pred_PP.append(predP)               
            pred_SS.append(predS)
                
        pred_DD = np.array(pred_DD).reshape(args['number_of_sampling'], len(test_dataset), 12000)
        pred_DD_mean = pred_DD.mean(axis=0)
        pred_DD_std = pred_DD.std(axis=0)  
        
        pred_PP = np.array(pred_PP).reshape(args['number_of_sampling'], len(test_dataset), 12000)
        pred_PP_mean = pred_PP.mean(axis=0)
        pred_PP_std = pred_PP.std(axis=0)      
        
        pred_SS = np.array(pred_SS).reshape(args['number_of_sampling'], len(test_dataset), 12000)
        pred_SS_mean = pred_SS.mean(axis=0)
        pred_SS_std = pred_SS.std(axis=0) 
    else:
        pred_DD_mean, pred_PP_mean, pred_SS_mean = model.predict_generator(generator=test_generator)
        pred_DD_mean = pred_DD_mean.reshape(pred_DD_mean.shape[0], pred_DD_mean.shape[1]) 
        pred_PP_mean = pred_PP_mean.reshape(pred_PP_mean.shape[0], pred_PP_mean.shape[1]) 
        pred_SS_mean = pred_SS_mean.reshape(pred_SS_mean.shape[0], pred_SS_mean.shape[1]) 
        
        pred_DD_std = np.zeros((pred_DD_mean.shape))
        pred_PP_std = np.zeros((pred_PP_mean.shape))
        pred_SS_std = np.zeros((pred_SS_mean.shape))

    plt_n = 0
    save_figs = f"{args['output_name']}/figures"

    if not os.path.exists(save_figs):
        os.makedirs(save_figs)

    for ts in range(pred_DD_mean.shape[0]):
        trace_name, p_sample, s_sample, data, is_earthquake = test_dataset.getinfo(ts)
        data = data.T
        if p_sample:
            p_sample = int(p_sample)
        if s_sample:
            s_sample = int(s_sample)

        matches, pick_errors, yh3=picker(args, pred_DD_mean[ts], pred_PP_mean[ts], pred_SS_mean[ts],
                                                       pred_DD_std[ts], pred_PP_std[ts], pred_SS_std[ts], p_sample, s_sample)
        
        _output_writter_test(args, trace_name, p_sample, s_sample, is_earthquake, test_writer, csvTst, matches, pick_errors)

        if plt_n < args['number_of_plots']:                          
            _plotter(trace_name,
                     p_sample,
                     s_sample,
                     data,
                     is_earthquake,
                     args,
                     save_figs,
                     pred_DD_mean[ts],
                     pred_PP_mean[ts],
                     pred_SS_mean[ts],
                     pred_DD_std[ts],
                     pred_PP_std[ts], 
                     pred_SS_std[ts],
                     matches)
        plt_n += 1

    # result = pd.read_csv(os.path.join(args['output_name'],'X_test_results.csv'))
    result = pd.read_csv(os.path.join(args['output_name'],'X_test_results.csv'))
    y_true = np.array(result['is_earthquake'] * 1)
    y_pred = np.array((result['number_of_detections'] > 0) * 1)

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
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('total number of tests '+str(len(args['test_dataset']))+'\n')            
        the_file.write('================== Other Parameters ========================='+'\n')        
        the_file.write('detection_threshold: '+str(args['detection_threshold'])+'\n')            
        the_file.write('P_threshold: '+str(args['P_threshold'])+'\n')
        the_file.write('S_threshold: '+str(args['S_threshold'])+'\n')
        the_file.write('number_of_plots: '+str(args['number_of_plots'])+'\n')
        the_file.write('confusion matrix: '+str(os.path.join(args['output_name'],"X_test_classification_report.csv"))+'\n')
        the_file.write('classification report: '+str(os.path.join(args['output_name'],"X_test_classification_report.csv"))+'\n')
 

def _output_writter_test(args,
                        trace_name, 
                        p_arrival, 
                        s_arrival,
                        is_earthquake, 
                        output_writer, 
                        csvfile, 
                        matches, 
                        pick_errors,
                        ):
    
    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    args: dic
        A dictionary containing all of the input parameters.    
 
    trace_name: str
        Trace name.  

    p_arrival: float
        p_arrival sample  

    s_arrival: float
        s_arrival sample 
    
    is_earthquake: bool
        Ground truth telling if signal is a earthquake
              
    output_writer: obj
        For writing out the detection/picking results in the CSV file.
        
    csvfile: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        Contains the information for the detected and picked event.  
      
    pick_errors: dic
        Contains prediction errors for P and S picks.          
        
    Returns
    --------  
    X_test_results.csv  
    
        
    """        
    
    
    numberOFdetections = len(matches)
    
    if numberOFdetections != 0: 
        D_prob =  matches[list(matches)[0]][1]
        D_unc = matches[list(matches)[0]][2]

        P_arrival = matches[list(matches)[0]][3]
        P_prob = matches[list(matches)[0]][4] 
        P_unc = matches[list(matches)[0]][5] 
        P_error = pick_errors[list(matches)[0]][0]
        
        S_arrival = matches[list(matches)[0]][6] 
        S_prob = matches[list(matches)[0]][7] 
        S_unc = matches[list(matches)[0]][8]
        S_error = pick_errors[list(matches)[0]][1]  
        
    else: 
        D_prob = None
        D_unc = None 

        P_arrival = None
        P_prob = None
        P_unc = None
        P_error = None
        
        S_arrival = None
        S_prob = None 
        S_unc = None
        S_error = None
       
    if P_unc:
        P_unc = round(P_unc, 3)


    output_writer.writerow([trace_name,
                            p_arrival,
                            s_arrival,
                            
                            numberOFdetections,
                            D_prob,
                            D_unc,    
                            
                            P_arrival, 
                            P_prob,
                            P_unc,                             
                            P_error,
                            
                            S_arrival, 
                            S_prob,
                            S_unc,
                            S_error,

                            is_earthquake
                            ]) 
    
    csvfile.flush()  


def _plotter(trace_name, p_sample, s_sample, data, is_earthquake, args, save_figs, yh1, yh2, yh3, yh1_std, yh2_std, yh3_std, matches):
    

    """ 
    
    Generates plots that draws the trace and the predication.

    Parameters
    ----------

    trace_name: str
        Trace name. 

    p_sample: int
        The P phase sample index

    s_sample: int
        The S phase sample index

    is_earthquake: bool
        Ground truth telling if signal is a earthquake

    args: dic
        A dictionary containing all of the input parameters. 

    save_figs: str
        Path to the folder for saving the plots. 

    yh1: 1D array
        Detection probabilities. 

    yh2: 1D array
        P arrival probabilities.   
      
    yh3: 1D array
        S arrival probabilities.  

    yh1_std: 1D array
        Detection standard deviations. 

    yh2_std: 1D array
        P arrival standard deviations.   
      
    yh3_std: 1D array
        S arrival standard deviations. 

    matches: dic
        Contains the information for the detected and picked event.  
          
        
    """
    
    spt = p_sample
    sst = s_sample

    predicted_P = []
    predicted_S = []
    if len(matches) >=1:
        for match, match_value in matches.items():
            if match_value[3]: 
                predicted_P.append(match_value[3])
            else:
                predicted_P.append(None)
                
            if match_value[6]:
                predicted_S.append(match_value[6])
            else:
                predicted_S.append(None)

    data = np.array(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(411)         
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    legend_properties = {'weight':'bold'}  
    plt.title(str(trace_name))
    plt.tight_layout()
    ymin, ymax = ax.get_ylim() 
    pl = None
    sl = None       
    ppl = None
    ssl = None  
    
    if is_earthquake:
        pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='P_Arrival')
        sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)     
                            
    ax = fig.add_subplot(412)   
    plt.plot(data[:, 1] , 'k')
    plt.tight_layout()                
    if is_earthquake:
        pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='P_Arrival')
        sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)    

    ax = fig.add_subplot(413) 
    plt.plot(data[:, 2], 'k')   
    plt.tight_layout()                
    if len(predicted_P) > 0:
        ymin, ymax = ax.get_ylim()
        for pt in predicted_P:
            if pt:
                ppl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Predicted_P_Arrival')
    if len(predicted_S) > 0:  
        for st in predicted_S: 
            if st:
                ssl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Predicted_S_Arrival')
                
    if ppl or ssl:    
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 

                
    ax = fig.add_subplot(414)
    x = np.linspace(0, data.shape[0], data.shape[0], endpoint=True)
    if args['estimate_uncertainty']:                               
        plt.plot(x, yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
        lowerD = yh1-yh1_std
        upperD = yh1+yh1_std
        plt.fill_between(x, lowerD, upperD, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')            
                            
        plt.plot(x, yh2, 'b--', alpha = 0.5, linewidth=1.5, label='P_probability')
        lowerP = yh2-yh2_std
        upperP = yh2+yh2_std
        plt.fill_between(x, lowerP, upperP, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')  
                                     
        plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
        lowerS = yh3-yh3_std
        upperS = yh3+yh3_std
        plt.fill_between(x, lowerS, upperS, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.ylim((-0.1, 1.1))
        plt.tight_layout()                
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)                 
    else:
        plt.plot(x, yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
        plt.plot(x, yh2, 'b--', alpha = 0.5, linewidth=1.5, label='P_probability')
        plt.plot(x, yh3, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
        plt.tight_layout()       
        plt.ylim((-0.1, 1.1))
        plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
    fig.savefig(os.path.join(save_figs, f'{trace_name}.png'))


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

