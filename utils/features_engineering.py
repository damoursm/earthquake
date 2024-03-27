import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

from config import FEATURES, FEATURES_SCALING


def add_random_feature(df):
    # TODO add random features for Standard Scaler
    df['random_feature'] = [
        random.uniform(FEATURES_SCALING['range'][0], FEATURES_SCALING['range'][1])
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


def assign_to_grid(lat, lon, lat_min, lon_min, grid_size=0.5):
    lat_idx = int((lat - lat_min) / grid_size)
    lon_idx = int((lon - lon_min) / grid_size)
    return (lat_idx, lon_idx)


def cap_outliers(data, outliers_scaler=None, up_limit=2, low_limit=2):
    data_cap = data.copy(deep=True)
    if outliers_scaler:
        new_scaler = outliers_scaler
    else:
        new_scaler = {}
        for col in data.columns:
            upper_limit = data[col].mean() + up_limit * data[col].std()
            lower_limit = data[col].mean() - low_limit * data[col].std()
            new_scaler[col] = {'upper_limit': upper_limit, 'lower_limit': lower_limit}
        # print(upper_limit)
        # print(lower_limit)

        # TODO corriger warning
        #  C:\Users\Math\anaconda3\lib\site-packages\pandas\core\indexing.py:965: SettingWithCopyWarning:
        #  A value is trying to be set on a copy of a slice from a DataFrame.
        #  Try using .loc[row_indexer,col_indexer] = value instead
        #  See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        #    self.obj[item] = s
        data_cap[col] = np.where(
            data[col] > upper_limit, upper_limit,
            np.where(
                data[col] < lower_limit, lower_limit, data[col]
            )
        )
        # print(data[col].describe())
        # print('-------------------------------------')

    return data_cap, new_scaler


def scale(data, features, scaling_params, scaler=None, outliers_limit=False, outliers_scaler=None, keep_raw=[]):
    if outliers_limit:  # This caps outliers only for the features that are used by the model
        data[features], outliers_scaler = cap_outliers(
            data[features],
            outliers_scaler=outliers_scaler,
            up_limit=outliers_limit,
            low_limit=outliers_limit
        )

    feats = list(outliers_scaler.keys()) if outliers_scaler else features  # TODO We need to use the same FEATURES as the ones used in training the model

    if scaler:  # Use the provided scaler
        new_scaler = False
        # print(scaler.data_min)
        # print(scaler.data_max)
        # scaler.fit(data)
        scaled = scaler.transform(data[feats])
    else:  # Create new scaler
        if scaling_params['name'] == 'MinMaxScaler':
            new_scaler = MinMaxScaler(feature_range=scaling_params['range']).fit(data[features])
            scaled = new_scaler.transform(data[features])
            # scaled = new_scaler.fit_transform(data[features])
        elif scaling_params['name'] == 'StandardScaler':
            new_scaler = StandardScaler().fit(data[features])
            scaled = new_scaler.transform(data[features])
            # scaled = new_scaler.fit_transform(data[features])

    # # Integer encode direction (pour si on doit encoder des string)
    # encoder = LabelEncoder()
    # values[:, 4] = encoder.fit_transform(values[:, 4])
    # # ensure all data is float
    # values = values.astype('float32')

    scaled_data = pd.DataFrame(scaled, index=data.index, columns=feats)
    scaled_data = scaled_data[features]

    for col in keep_raw:
        scaled_data[col] = data[col]

    return scaled_data, new_scaler, outliers_scaler


def data_enrich(df_meta):
    # Make feature lat-long grid
    lat_min = df_meta['station_latitude_deg'].min()
    lon_min = df_meta['station_longitude_deg'].min()
    df_meta['grid_cell'] = df_meta.apply(
        lambda x: assign_to_grid(x['station_latitude_deg'], x['station_longitude_deg'], lat_min, lon_min),
        axis=1
    )
    # Other data enrich...

    return df_meta
