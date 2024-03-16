import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import torch


def plot_trace_prediction(trace_name, data, p_samples, s_samples, detection_probs, p_arrival_probs, s_arrival_probs, save_figs):

    """ 

    Generates plots of detected events waveforms, output predictions, and picked arrival times.

    Parameters
    ----------
    trace_name : str
        Trace name.

    data: NumPy array
        3 component raw waveform. Shape should be 3 x nbSamples

    p_samples: int
        Sample where P phase happens. 

    s_samples: 1D array
        Sample where S phase happens. 

    detection_probs: 1D array
        Detection probabilities. 

    p_arrival_probs: 1D array
        P arrival probabilities.    
        
    p_arrival_probs: 1D array
        S arrival probabilities. 

    save_figs: str
        Path to the folder for saving the plots.   
            
    """   
        
    ########################################## ploting only in time domain
    fig = plt.figure(constrained_layout=True)
    widths = [1]
    heights = [1.6, 1.6, 1.6, 2.5]
    spec5 = fig.add_gridspec(ncols=1, nrows=4, width_ratios=widths,
                            height_ratios=heights)
    
    ax = fig.add_subplot(spec5[0, 0])         
    plt.plot(data[0, :], 'k')
    x = np.arange(data.shape[1])
    plt.xlim(0, data.shape[1])           
    
    plt.ylabel('Amplitude\nCounts')                     
                
    plt.rcParams["figure.figsize"] = (8,6)
    legend_properties = {'weight':'bold'}  
    plt.title(f'Trace Name: {trace_name}')
    
    pl = sl = None        
    if len(p_samples) > 0:
        ymin, ymax = ax.get_ylim()
        for ipt, pt in enumerate(p_samples):
            if pt and ipt == 0:
                pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
            elif pt and ipt > 0:
                pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                
    if len(s_samples) > 0: 
        for ist, st in enumerate(s_samples): 
            if st and ist == 0:
                sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
            elif st and ist > 0:
                sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                
    if pl or sl:    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        custom_lines = [Line2D([0], [0], color='k', lw=0),
                        Line2D([0], [0], color='c', lw=2),
                        Line2D([0], [0], color='m', lw=2)]
        plt.legend(custom_lines, ['E', 'Picked P', 'Picked S'], 
                    loc='center left', bbox_to_anchor=(1, 0.5), 
                    fancybox=True, shadow=True)
                                        
    ax = fig.add_subplot(spec5[1, 0])   
    plt.plot(data[1, :] , 'k')
    plt.xlim(0, data.shape[1])            
    plt.ylabel('Amplitude\nCounts')           
                
    if len(p_samples) > 0:
        ymin, ymax = ax.get_ylim()
        for ipt, pt in enumerate(p_samples):
            if pt and ipt == 0:
                pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
            elif pt and ipt > 0:
                pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                
    if len(s_samples) > 0: 
        for ist, st in enumerate(s_samples): 
            if st and ist == 0:
                sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
            elif st and ist > 0:
                sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)

    if pl or sl:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        custom_lines = [Line2D([0], [0], color='k', lw=0),
                        Line2D([0], [0], color='c', lw=2),
                        Line2D([0], [0], color='m', lw=2)]
        plt.legend(custom_lines, ['N', 'Picked P', 'Picked S'], 
                    loc='center left', bbox_to_anchor=(1, 0.5), 
                    fancybox=True, shadow=True)
                        
    ax = fig.add_subplot(spec5[2, 0]) 
    plt.plot(data[2, :], 'k') 
    plt.xlim(0, data.shape[1])                    
    plt.ylabel('Amplitude\nCounts')
           
    ax.set_xticks([])
                
    if len(p_samples) > 0:
        ymin, ymax = ax.get_ylim()
        for ipt, pt in enumerate(p_samples):
            if pt and ipt == 0:
                pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2, label='Picked P')
            elif pt and ipt > 0:
                pl = plt.vlines(int(pt), ymin, ymax, color='c', linewidth=2)
                
    if len(s_samples) > 0:
        for ist, st in enumerate(s_samples): 
            if st and ist == 0:
                sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2, label='Picked S')
            elif st and ist > 0:
                sl = plt.vlines(int(st), ymin, ymax, color='m', linewidth=2)
                
    if pl or sl:    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        custom_lines = [Line2D([0], [0], color='k', lw=0),
                        Line2D([0], [0], color='c', lw=2),
                        Line2D([0], [0], color='m', lw=2)]
        plt.legend(custom_lines, ['Z', 'Picked P', 'Picked S'], 
                    loc='center left', bbox_to_anchor=(1, 0.5), 
                    fancybox=True, shadow=True)       
                
    ax = fig.add_subplot(spec5[3, 0])
    x = np.linspace(0, data.shape[1], data.shape[1], endpoint=True)
    
    #Plot predictions
    if torch.is_tensor(detection_probs):
        plt.plot(x, detection_probs, '--', color='g', alpha = 0.5, linewidth=1.5, label='Earthquake')

    if torch.is_tensor(p_arrival_probs):
        plt.plot(x, p_arrival_probs, '--', color='b', alpha = 0.5, linewidth=1.5, label='P_arrival')

    if torch.is_tensor(s_arrival_probs):
        plt.plot(x, s_arrival_probs, '--', color='r', alpha = 0.5, linewidth=1.5, label='S_arrival')

    plt.tight_layout()       
    plt.ylim((-0.1, 1.1)) 
    plt.xlim(0, data.shape[1])                                            
    plt.ylabel('Probability') 
    plt.xlabel('Sample')  
    plt.legend(loc='lower center', bbox_to_anchor=(0., 1.17, 1., .102), ncol=3, mode="expand",
                prop=legend_properties, borderaxespad=0., fancybox=True, shadow=True)
    plt.yticks(np.arange(0, 1.1, step=0.2))
    axes = plt.gca()
    axes.yaxis.grid(color='lightgray')
        
    fig.tight_layout()
    fig.savefig(os.path.join(save_figs, str(trace_name).replace(':', '-')+'.png')) 
    plt.close(fig)
    plt.clf()
