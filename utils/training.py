from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from nn import cnn
from config import features_list


def train_validate(hyperparams, train_data, test_data):
    if hyperparams['name'] == 'Random Forest':
        model = train_rf(hyperparams, train_data)

    metric = validate(model, test_data, hyperparams['metric'])
    return metric


def train_rf(hyperparams, train_data):
    X_train, y_train = train_data[features_list], train_data['source']
    clf = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_cnn(hyperparams, train_data):
    cnn(hyperparams, train_data)


def validate(model, test_data, metric_nm):
    preds = model.predict(test_data[features_list])
    if metric_nm == 'auc':
        metric_value = roc_auc_score(test_data['source'], preds)

    return metric_value
