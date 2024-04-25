from __future__ import print_function
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
from keras import layers
from keras.layers import Input
from keras.models import load_model, Sequential
from keras.optimizers.legacy import Adam

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from dotenv import dotenv_values
import shutil

from utils.dataloaders.InstanceDataset import InstanceDataset
from utils.preprocess_data import shuffle_events
from utils.keras_detection_trainer import trainer, tester, _lr_schedule

def main():
    env_values = dotenv_values(".env")

    event_hdf5_file = os.environ.get('SLURM_TMPDIR_EVENT_HDF5_FILE', env_values["EVENT_HDF5_FILE"])
    event_metadata_file = os.environ.get('SLURM_TMPDIR_EVENT_METADATA_FILE', env_values["EVENT_METADATA_FILE"])
    noise_hdf5_file = os.environ.get('SLURM_TMPDIR_NOISE_HDF5_FILE', env_values["NOISE_HDF5_FILE"])
    noise_metadata_file = os.environ.get('SLURM_TMPDIR_NOISE_METADATA_FILE', env_values["NOISE_METADATA_FILE"])

    temp_dir = os.environ.get('SLURM_TMPDIR', env_values["TEMP_DIR"])

    final_output_dir = env_values["FINAL_OUTPUT_DIR"]

    best_model_name='best_model.h5'

    temp_dir_output = f"{temp_dir}/output"

    if not os.path.exists(temp_dir_output):
        os.makedirs(temp_dir_output)

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
    target_type="binary"
    norm_mode='std'

    train_dataset = InstanceDataset(event_hdf5_file, shuffled_event_metadata_file, noise_hdf5_file, shuffled_noise_metadata_file, target_type=target_type, split_index=0, split_percentage=split_percentage, padding_type=padding_type, start_padding_value=start_padding_value, end_padding_value=end_padding_value, target_phase=target_phase, phase_padding=phase_padding, remove_empty_phase=remove_empty_phase, norm_mode=norm_mode)
    val_dataset = InstanceDataset(event_hdf5_file, shuffled_event_metadata_file, noise_hdf5_file, shuffled_noise_metadata_file, target_type=target_type, split_index=1, split_percentage=split_percentage, padding_type=padding_type, start_padding_value=start_padding_value, end_padding_value=end_padding_value, target_phase=target_phase, phase_padding=phase_padding, remove_empty_phase=remove_empty_phase, norm_mode=norm_mode)
    test_dataset = InstanceDataset(event_hdf5_file, shuffled_event_metadata_file, noise_hdf5_file, shuffled_noise_metadata_file, target_type=target_type, split_index=2, split_percentage=split_percentage, padding_type=padding_type, start_padding_value=start_padding_value, end_padding_value=end_padding_value, target_phase=target_phase, phase_padding=phase_padding, remove_empty_phase=remove_empty_phase, norm_mode=norm_mode)

    print(f"Dataset size: Train={len(train_dataset)} - Val={len(val_dataset)} - Test={len(test_dataset)}")

    data, detection, p_target, s_target, trace_name = train_dataset[0]
    print(f"Single Earthquake Data shape: {data.shape}")
    print(f"Single Earthquake detection: {detection}")
    print(f"Single Earthquake p Target: {p_target}")
    print(f"Single Earthquake s Target: {s_target}")
    print(f"Single Earthquake trace name: {trace_name}")

    data, detection, p_target, s_target, trace_name = train_dataset[len(train_dataset) - 1]
    print(f"Single Earthquake Data shape: {data.shape}")
    print(f"Single Earthquake detection: {detection}")
    print(f"Single Earthquake p Target: {p_target}")
    print(f"Single Earthquake s Target: {s_target}")
    print(f"Single Earthquake trace name: {trace_name}")

    print("############################ Model ############################")

    # Define the input shape
    input_shape = (12000, 3)
    epochs=30
    batch_size=200

    model = _get_model(input_shape=input_shape)
    model.summary()

    model.compile(optimizer=Adam(learning_rate=_lr_schedule(0)), loss='binary_crossentropy', metrics=['accuracy'])

    print("############################ Training ############################")

    trainer(train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            output_name=temp_dir_output,
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

    model_path = f'{temp_dir_output}/{best_model_name}'
    model = load_model(model_path)

    model.compile(optimizer=Adam(learning_rate=_lr_schedule(0)), loss='binary_crossentropy', metrics=['accuracy'])
    
    tester(test_dataset=test_dataset,
           model=model,
           output_name=temp_dir_output,
           batch_size=batch_size
    )


    shutil.copytree(temp_dir_output, final_output_dir, dirs_exist_ok=True)

    print("########################################################")


def _get_model(input_shape):
    model = Sequential([
        layers.Reshape(input_shape, input_shape=input_shape),
        layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(1),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

if __name__ == '__main__':
    main()