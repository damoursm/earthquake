from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from keras.layers import Input
from keras.models import load_model
from keras.optimizers.legacy import Adam

from nn.eq_transformer import EqModel, f1, _lr_schedule, SeqSelfAttention, LayerNormalization, FeedForward

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from dotenv import dotenv_values

from utils.dataloaders.InstanceDataset import InstanceDataset
from utils.preprocess_data import shuffle_events
from utils.keras_generic_trainer import trainer, tester


def main():
    env_values = dotenv_values(".env")

    event_hdf5_file = os.environ.get('SLURM_TMPDIR_EVENT_HDF5_FILE', env_values["EVENT_HDF5_FILE"])
    event_metadata_file = os.environ.get('SLURM_TMPDIR_EVENT_METADATA_FILE', env_values["EVENT_METADATA_FILE"])
    noise_hdf5_file = os.environ.get('SLURM_TMPDIR_NOISE_HDF5_FILE', env_values["NOISE_HDF5_FILE"])
    noise_metadata_file = os.environ.get('SLURM_TMPDIR_NOISE_METADATA_FILE', env_values["NOISE_METADATA_FILE"])

    temp_dir = os.environ.get('SLURM_TMPDIR', env_values["TEMP_DIR"])

    final_output_dir = env_values["FINAL_OUTPUT_DIR"]

    best_model_name='best_model.h5'

    print("############################ Data set ############################")

    shuffled_event_metadata_file, shuffled_noise_metadata_file = shuffle_events(
        event_metadata_file=event_metadata_file,
        noise_metadata_file=noise_metadata_file,
        random_state=9,
        output_event_metadata_file=f"{temp_dir}/shuffled_event_metadata.csv",
        output_noise_metadata_file=f"{temp_dir}/shuffled_noise_metadata.csv",
        balance=True,
        remove_empty_phase=True)

    split_percentage=[0.80, 0.10, 0.10]

    print(f"Earthquake hdf5 file: {event_hdf5_file}")
    print(f"Earthquake metadata file: {event_metadata_file}")
    print(f"Noise hdf5 file: {noise_hdf5_file}")
    print(f"Noise metadata file: {noise_metadata_file}")
    print(f"Earthquake metadata file (shuffled): {shuffled_event_metadata_file}")
    print(f"Noise metadata file (shuffled): {shuffled_noise_metadata_file}")
    
    padding_type="percentage"
    start_padding_value=0
    end_padding_value=1.4
    target_phase=True
    phase_padding=20
    remove_empty_phase=True #Keep it to true as when a phase is missing, the box will be malformed
    target_type="box_square"
    #target_type="box_square"
    norm_mode='std'

    train_dataset = InstanceDataset(event_hdf5_file, shuffled_event_metadata_file, noise_hdf5_file, shuffled_noise_metadata_file, target_type=target_type, split_index=0, split_percentage=split_percentage, padding_type=padding_type, start_padding_value=start_padding_value, end_padding_value=end_padding_value, target_phase=target_phase, phase_padding=phase_padding, remove_empty_phase=remove_empty_phase, norm_mode=norm_mode)
    val_dataset = InstanceDataset(event_hdf5_file, shuffled_event_metadata_file, noise_hdf5_file, shuffled_noise_metadata_file, target_type=target_type, split_index=1, split_percentage=split_percentage, padding_type=padding_type, start_padding_value=start_padding_value, end_padding_value=end_padding_value, target_phase=target_phase, phase_padding=phase_padding, remove_empty_phase=remove_empty_phase, norm_mode=norm_mode)
    test_dataset = InstanceDataset(event_hdf5_file, shuffled_event_metadata_file, noise_hdf5_file, shuffled_noise_metadata_file, target_type=target_type, split_index=2, split_percentage=split_percentage, padding_type=padding_type, start_padding_value=start_padding_value, end_padding_value=end_padding_value, target_phase=target_phase, phase_padding=phase_padding, remove_empty_phase=remove_empty_phase, norm_mode=norm_mode)

    print(f"Dataset size: Train={len(train_dataset)} - Val={len(val_dataset)} - Test={len(test_dataset)}")

    data, detection, p_target, s_target, trace_name = train_dataset[0]
    print(f"Single Earthquake Data shape: {data.shape}")
    print(f"Single Earthquake detection shape: {detection.shape}")
    print(f"Single Earthquake p Target shape: {p_target.shape}")
    print(f"Single Earthquake s Target shape: {s_target.shape}")
    print(f"Single Earthquake trace name: {trace_name}")

    data, detection, p_target, s_target, trace_name = train_dataset[len(train_dataset) - 1]
    print(f"Single Earthquake Data shape: {data.shape}")
    print(f"Single Earthquake detection shape: {detection.shape}")
    print(f"Single Earthquake p Target shape: {p_target.shape}")
    print(f"Single Earthquake s Target shape: {s_target.shape}")
    print(f"Single Earthquake trace name: {trace_name}")

    print("############################ Model ############################")

    input_dimention=(12000, 3)

    inp = Input(shape=input_dimention, name='input') 
    model, batch_size, epochs, loss_types, loss_weights = _get_model(len(train_dataset), inp)
    model.summary()

    print("############################ Training ############################")

    trainer(train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            output_name=final_output_dir,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            monitor='val_loss',
            patience=12,
            gpuid=None,
            gpu_limit=None,
            use_multiprocessing=True,
            best_model_name=best_model_name)
    
    print("############################ Testing ############################")

    model_path = f'{final_output_dir}/{best_model_name}'
    model = load_model(model_path, custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                                         'FeedForward': FeedForward,
                                                         'LayerNormalization': LayerNormalization, 
                                                         'f1': f1                                                                            
                                                         })

    model.compile(loss = loss_types,
                  loss_weights = loss_weights,           
                  optimizer = Adam(learning_rate=_lr_schedule(0)),
                  metrics = [f1])
    
    tester(test_dataset=test_dataset,
           model=model,
           output_name=final_output_dir,
           detection_threshold=0.5,
           P_threshold=0.3,
           S_threshold=0.3,
           estimate_uncertainty=False,
           number_of_sampling=5,
           number_of_plots=100,
           loss_weights=loss_weights,
           loss_types=loss_types,
           batch_size=batch_size
    )
    print("########################################################")


def _get_model(train_data_size, input):
    nb_filters=[8, 16, 16, 32, 32, 64, 64, 96]
    kernel_size=[11, 9, 7, 7, 5, 5, 3, 3]
    endcoder_depth=8
    decoder_depth=8
    cnn_blocks=5
    lstm_blocks=2
    kernel_padding='same'
    activation = 'relu'         
    drop_rate=0.2
    loss_weights=[0.20, 0.40, 0.4]
    loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy']
    batch_size = 200
    epochs = 30

    return EqModel(nb_filters=nb_filters,
              kernel_size=kernel_size,
              padding=kernel_padding,
              activationf=activation,
              endcoder_depth=endcoder_depth,
              decoder_depth=decoder_depth,
              cnn_blocks=cnn_blocks,
              BiLSTM_blocks=lstm_blocks,
              drop_rate=drop_rate, 
              loss_weights=loss_weights,
              loss_types=loss_types,
              kernel_regularizer=1e-3,
              bias_regularizer=1e-3
               )(input), batch_size, epochs, loss_types, loss_weights

if __name__ == '__main__':
    main()