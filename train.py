from utils.dataloaders.EventDetectionInstanceDataset import EventDetectionInstanceDataset
from utils.generic_trainer import train_detection_only
from utils.plot import plot_error_and_accuracy

from nn.cnn import CNN
from torch import nn

from utils.correct_counter import correct

import shutil
import os

from dotenv import dotenv_values

def main():
    env_values = dotenv_values(".env")

    event_hdf5_file = os.environ.get('SLURM_TMPDIR_EVENT_HDF5_FILE', env_values["EVENT_HDF5_FILE"])
    event_metadata_file = os.environ.get('SLURM_TMPDIR_EVENT_METADATA_FILE', env_values["EVENT_METADATA_FILE"])
    noise_hdf5_file = os.environ.get('SLURM_TMPDIR_NOISE_HDF5_FILE', env_values["NOISE_HDF5_FILE"])
    noise_metadata_file = os.environ.get('SLURM_TMPDIR_NOISE_METADATA_FILE', env_values["NOISE_METADATA_FILE"])

    temp_dir = os.environ.get('SLURM_TMPDIR', env_values["TEMP_DIR"])

    final_output_dir = env_values["FINAL_OUTPUT_DIR"]

    split_percentage = [0.8, 0.1, 0.1]

    print("############################ Data set ############################")

    print(f"Earthquake hdf5 file: {event_hdf5_file}")
    print(f"Earthquake metadata file: {event_metadata_file}")
    print(f"Noise hdf5 file: {noise_hdf5_file}")
    print(f"Noise metadata file: {noise_metadata_file}")

    train_dataset = EventDetectionInstanceDataset(event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, "binary", split_index=0, split_percentage=split_percentage, padding_type="sample", padding_value=100)
    val_dataset = EventDetectionInstanceDataset(event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, "binary", split_index=1, split_percentage=split_percentage, padding_type="sample", padding_value=100)
    test_dataset = EventDetectionInstanceDataset(event_hdf5_file, event_metadata_file, noise_hdf5_file, noise_metadata_file, "binary", split_index=2, split_percentage=split_percentage, padding_type="sample", padding_value=100)

    print(f"Dataset size: Train={len(train_dataset)} - Val={len(val_dataset)} - Test={len(test_dataset)}")

    data, target = train_dataset[0]
    print(f"Single Earthquake Data shape: {data.shape}")
    print(f"Single Earthquake Target: {target}")

    data, target = train_dataset[len(train_dataset) - 1]
    print(f"Single Noise Data shape: {data.shape}")
    print(f"Single Noise Target: {target}")

    print("############################ Model ############################")

    model = CNN(
        input_channels=3,
        conv_channels= [
            8, 16, 32, 64, 128
        ], kernel_sizes=[
            11, 9, 7, 5, 3
        ], mlp_layers=[
            128, 64, 32, 2
        ],
        dropout=0.4
    )

    print(model)

    print("############################ Training ############################")

    loss = nn.CrossEntropyLoss()

    e, a, model_path, best_val_accuracy, monitor = train_detection_only(
        train_dataset, val_dataset, model, loss, correct, batch_size=64, epochs=20, temp_dir=temp_dir
    )

    plot_error_and_accuracy(e, a, final_output_dir)
    shutil.copy(model_path, final_output_dir)
    monitor.save_plots(final_output_dir)


    print("########################################################")

    # with h5py.File("/project/def-sponsor00/earthquake/data/instance/Instance_events_counts.hdf5", 'r') as f:
    #     print(f['data'])
    #     print(f['data'].keys())
    #     print(f['data']['11030611.IV.OFFI..HH'])
    #     print(f['data']['11030611.IV.OFFI..HH'][2, :15])

    # a = np.array([0, 1, 2])
    # print(a**2)



if __name__ == '__main__':
    main()