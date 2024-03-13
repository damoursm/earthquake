import skopt

INSTANCE_EVENTS_FILENAME = 'Instance_sample_dataset/data/Instance_events_counts_10k.hdf5'
INSTANCE_NOISE_FILENAME = 'Instance_sample_dataset/data/Instance_noise_1k.hdf5'
INSTANCE_EVENTS_METADATA_FILENAME = 'Instance_sample_dataset/metadata/metadata_Instance_events_10k.csv'
INSTANCE_NOISE_METADATA_FILENAME = 'Instance_sample_dataset/metadata/metadata_Instance_noise_1k.csv'

FEATURES = []
FEATURES_SCALE = []
FEATURES_IMPORTANCE = []
MODEL = 'model_name'

HYPERPARAMETERS = [
    skopt.space.Real(10, 70, name='t_score_feats_filter'),
    skopt.space.Categorical([70], name='t_score_feats_filter'),
]