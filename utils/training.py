from dotenv import dotenv_values
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import shutil
from torch import nn
from dotenv import dotenv_values

from config import features_list
from nn.cnn import CNN

from utils.correct_counter import correct
from utils.dataloaders.EventDetectionInstanceDataset import EventDetectionInstanceDataset
from utils.generic_trainer import train_detection_only
from utils.plot import plot_error_and_accuracy



def train_validate(hyperparams, train_data, test_data):
    if hyperparams['name'] == 'Random Forest':
        model, artifacts = train_rf(hyperparams, train_data)
        metrics, artifacts_ = validate(model, test_data, hyperparams['metric'])
        artifacts += artifacts_

    elif hyperparams['name'] == 'cnn_elisee':
        model, metrics, artifacts = train_cnn(hyperparams)

    return model, metrics, artifacts


def train_rf(hyperparams, train_data):
    X_train, y_train = train_data[features_list], train_data['source']
    clf = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf, []


def train_cnn(hyperparams):
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

    # TODO rajouter hyperparametres
    model = CNN(
        input_channels=3,
        conv_channels=[8, 16, 32, 64, 128],
        kernel_sizes=[11, 9, 7, 5, 3],
        mlp_layers=[128, 64, 32, 2],
        dropout=hyperparams['dropout'],
    )

    print(model)

    print("############################ Training ############################")

    loss = nn.CrossEntropyLoss()

    e, a, best_model_path, best_val_accuracy, monitor = train_detection_only(
        train_dataset,
        val_dataset,
        model,
        loss,
        correct,
        batch_size=int(hyperparams['batch_size']),
        epochs=hyperparams['epochs'],
        temp_dir=temp_dir
    )

    plot_error_and_accuracy(e, a, final_output_dir)
    shutil.copy(best_model_path, final_output_dir)
    monitor.save_plots(final_output_dir)

    return best_model_path, {'metric_eval': best_val_accuracy}, []


def validate(model, test_data, metric_nm):
    metrics = {}
    preds = model.predict(test_data[features_list])
    if metric_nm == 'auc':
        metrics['metric_eval'] = roc_auc_score(test_data['source'], preds)
    metrics['auc'] = roc_auc_score(test_data['source'], preds)
    metrics['accuracy'] = accuracy_score(test_data['source'], preds)
    metrics['f1'] = f1_score(test_data['source'], preds)
    metrics['precision'] = precision_score(test_data['source'], preds)
    metrics['recall'] = recall_score(test_data['source'], preds)

    # file_path = os.path.join('mlruns', self.exp_id, 'output/figures/Training_report.png')
    artifacts = []

    return metrics, artifacts


def test(model, test_data, metric_nm):
    preds = model.predict(test_data[features_list])
    if metric_nm == 'auc':
        metric_value = roc_auc_score(test_data['source'], preds)

    return metric_value
