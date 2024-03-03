import skopt

FEATURES = []
FEATURES_SCALE = []
FEATURES_IMPORTANCE = []
MODEL = 'model_name'

HYPERPARAMETERS = [
    skopt.space.Real(10, 70, name='t_score_feats_filter'),
    skopt.space.Categorical([70], name='t_score_feats_filter'),
]