import skopt


EXPERIMENT_NAME = ''

INSTANCE_EVENTS_FILENAME = 'Instance_sample_dataset/data/Instance_events_counts_10k.hdf5'
INSTANCE_NOISE_FILENAME = 'Instance_sample_dataset/data/Instance_noise_1k.hdf5'
INSTANCE_EVENTS_METADATA_FILENAME = 'Instance_sample_dataset/metadata/metadata_Instance_events_10k.csv'
INSTANCE_NOISE_METADATA_FILENAME = 'Instance_sample_dataset/metadata/metadata_Instance_noise_1k.csv'

SIGNALS_PREDS = 'data/signals_preds.csv'  # TODO faire CSV des signals predictions
BALANCE_METADATA = ['train', 'validate', 'test']

feat_eng_num = [('outliers_filter', 3), ('MinMaxScaler', (0, 1))]
feat_eng_num_minmax = [('MinMaxScaler', (0, 1))]

FEATURES = {
    'numerical': {
        'station_latitude_deg': feat_eng_num_minmax,
        'station_longitude_deg': feat_eng_num_minmax,
        'station_elevation_m': feat_eng_num_minmax,
        'station_vs_30_mps': feat_eng_num_minmax,
        'trace_dt_s': feat_eng_num_minmax,
        'trace_npts': feat_eng_num_minmax,
        'trace_E_median_counts': feat_eng_num_minmax,
        'trace_N_median_counts': feat_eng_num_minmax,
        'trace_Z_median_counts': feat_eng_num_minmax,
        'trace_E_mean_counts': feat_eng_num_minmax,
        'trace_N_mean_counts': feat_eng_num_minmax,
        'trace_Z_mean_counts': feat_eng_num_minmax,
        'trace_E_min_counts': feat_eng_num_minmax,
        'trace_N_min_counts': feat_eng_num_minmax,
        'trace_Z_min_counts': feat_eng_num_minmax,
        'trace_E_max_counts': feat_eng_num_minmax,
        'trace_N_max_counts': feat_eng_num_minmax,
        'trace_Z_max_counts': feat_eng_num_minmax,
        'trace_E_rms_counts': feat_eng_num_minmax,
        'trace_N_rms_counts': feat_eng_num_minmax,
        'trace_Z_rms_counts': feat_eng_num_minmax,
        'trace_E_lower_quartile_counts': feat_eng_num_minmax,
        'trace_N_lower_quartile_counts': feat_eng_num_minmax,
        'trace_Z_lower_quartile_counts': feat_eng_num_minmax,
        'trace_E_upper_quartile_counts': feat_eng_num_minmax,
        'trace_N_upper_quartile_counts': feat_eng_num_minmax,
        'trace_Z_upper_quartile_counts': feat_eng_num_minmax,
        'trace_E_spikes': feat_eng_num_minmax,
        'trace_N_spikes': feat_eng_num_minmax,
        'trace_Z_spikes': feat_eng_num_minmax,
        # 'trace_GPD_P_number': feat_eng_num_minmax,
        # 'trace_GPD_S_number': feat_eng_num_minmax,
        # 'trace_EQT_number_detections': feat_eng_num_minmax,
        # 'trace_EQT_P_number': feat_eng_num_minmax,
        # 'trace_EQT_S_number': feat_eng_num_minmax,

        # 'station_distance_to_coast_km': feat_eng_num_minmax,
        # 'station_distance_to_nearest_tectonic_plate_boundary_km': feat_eng_num_minmax,
        # 'station_distance_to_nearest_active_fault_km': feat_eng_num_minmax,
        # 'station_distance_to_nearest_active_volcano_km': feat_eng_num_minmax,
    },
    'categorical': {  # TODO add categorical features by doing one hot encoding
        # 'station_network_code': 'ohe',
        # 'station_code': 'ohe',
        # 'station_channels': 'ohe',
        # 'station_vs_30_detail': 'ohe',
        # 'trace_start_time': 'ohe',
        'station_grid_cell': 'ohe',
        # # 'station_country': 'ohe',
    }
}

features_list = list(FEATURES['numerical'].keys())

FEATURES_SCALING = {
    'name': 'MinMaxScaler',
    'range': (0, 1),
}
# FEATURES_SCALING = {
#     'name': 'StandardScaler',
# }

NB_ITER = 10
METRIC_EVAL = 'AUC'
RETRAIN_FOR_TEST = False

# TODO rajouter hyperparametres dept et width pour CNN
HYPERPARAMETERS = [
    skopt.space.Categorical(['cnn_elisee'], name='name'),
    skopt.space.Real(0, 0.9, name='dropout'),
    skopt.space.Integer(2, 3, name='epochs'),
    skopt.space.Categorical([32, 64, 128], name='batch_size'),
    skopt.space.Categorical(['accuracy'], name='metric'),
]

HYPERPARAMETERS = [
    skopt.space.Categorical(['Random Forest'], name='name'),
    skopt.space.Integer(100, 1000, name='n_estimators'),
    skopt.space.Integer(10, 100, name='max_depth'),
    skopt.space.Integer(2, 10, name='min_samples_leaf'),
    skopt.space.Integer(1, 10, name='max_leaf_nodes'),
    skopt.space.Categorical(['auto'], name='max_features'),
    skopt.space.Categorical([True], name='bootstrap'),
    skopt.space.Categorical(['auc'], name='metric'),
]
