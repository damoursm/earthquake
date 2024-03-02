import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random

from config import FEATURES, FEATURES_SCALE


def add_random_feature(df):
    df['random_feature'] = [
        random.uniform(FEATURES_SCALE['range'][0], FEATURES_SCALE['range'][1])
        for _ in range(len(df))
    ]
    return df


def t_score_filter(data, filter_perc, target='weighted_target_scaled'):
    t_scores = {}
    num_features = FEATURES
    data_pos = data[data[target] >= 1]
    data_neg = data[data[target] < 1]
    for feat in num_features:
        avg_pos = data_pos[feat].mean()
        avg_neg = data_neg[feat].mean()

        std_pos = data_pos[feat].std()
        std_neg = data_neg[feat].std()

        n_pos = data_pos[feat].count()
        n_neg = data_neg[feat].count()

        if avg_pos == avg_neg and std_pos == 0.0 and std_neg == 0.0:
            t_scores[feat] = 0
        else:
            t_scores[feat] = abs(avg_pos - avg_neg) / np.sqrt(std_pos**2/n_pos + std_neg**2/n_neg)

    threshold = np.nanpercentile(list(t_scores.values()), filter_perc)
    imp_features = {
        feat: score
        for feat, score in t_scores.items()
        if score >= threshold
    }

    return list(imp_features)


def shap_feat_imp(model, X_train):
    shap_values = []
    shap_values_abs = []

    baseline_prediction = model.predict(X_train)

    for feat in X_train.columns:
        permuted_observation = X_train.copy()
        permuted_observation.loc[:, feat] = 0  # Permute the i-th feature

        permuted_prediction = model.predict(permuted_observation)

        shap_values.append(np.mean(baseline_prediction - permuted_prediction))
        shap_values_abs.append(np.mean(abs(baseline_prediction - permuted_prediction)))

    return shap_values, shap_values_abs


def plot_shap(shap_df):
    plt.figure(figsize=(10, 6))
    plt.barh(shap_df['Feature'], shap_df['SHAP Value'], color='skyblue')
    plt.xlabel('Valeur SHAP')
    plt.ylabel('Features')
    plt.title('Valeurs SHAP pour chaque caractéristique')
    plt.show()


def make_shap(train_data, model, features, verbose=False):
    shap_values, shap_values_abs = shap_feat_imp(model, train_data[features])
    print("SHAP important features:", shap_values)
    shap_df = pd.DataFrame({'Feature': list(features), 'SHAP Value': shap_values})
    shap_df = shap_df.sort_values(by='SHAP Value', ascending=False)
    if verbose:
        plot_shap(shap_df)
    shap_abs_df = pd.DataFrame({'Feature': list(features), 'SHAP Value': shap_values_abs})
    shap_abs_df = shap_abs_df.sort_values(by='SHAP Value', ascending=False)
    if verbose:
        plot_shap(shap_abs_df)
    forest_importances = pd.Series(shap_values_abs, index=features).sort_values(ascending=False)
    forest_importances.to_dict().items()

    imp_features = []
    imp_rand_feat = forest_importances.to_dict()['random']
    for k, v in forest_importances.to_dict().items():
        if v > 0 and v > imp_rand_feat:
            imp_features.append(k)

    return imp_features


def PCA():
    pass