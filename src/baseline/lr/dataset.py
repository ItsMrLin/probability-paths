import pandas as pd
import numpy as np
from scipy.special import logit, expit
from sklearn.preprocessing import OneHotEncoder
import logging

import constants as C

class Dataset:
    def __init__(self):
        self.train_raw = pd.read_csv(self.config.TRAIN_PATH)
        self.test_raw = pd.read_csv(self.config.TEST_PATH)
        self.train_features = None
        self.test_features = None
        self.train_ids = self.get_ids("train")
        self.test_ids = self.get_ids("test")

    def get_ids(self, split="train"):
        if split == "train":
            df = self.train_raw
        else:
            df = self.test_raw
        return sorted(df[self.config.ID].unique())

    def get_label(self, t, split="train"):
        if split == "train":
            df = self.train_raw
        else:
            df = self.test_raw
        label = df.query(f"time=={t}") \
                  .sort_values(self.config.ID)[self.config.PREDICTION] \
                  .values
        label_logit = logit(np.clip(label, C.EPSILON, 1-C.EPSILON)) 
        return label_logit

    def get_features(self, split="train"):
        if split == "train":
            if self.train_features is None:
                df = self.train_raw.copy()
                self.train_features = self.create_features(df)
            assert len(self.train_features) == len(self.train_ids)
            features = self.train_features
        else:
            if self.test_features is None:
                df = self.test_raw.copy()
                self.test_features = self.create_features(df)
            assert len(self.test_features) == len(self.test_ids)
            features = self.test_features
        return features.drop(columns=self.config.ID)


class NBADataset(Dataset):
    def __init__(self, **kwargs):
        self.config = C.NBAConfig
        super().__init__(**kwargs)
    
    def create_features(self, df):
        logging.info("Generating features for training dataset...")
        config = self.config
        IDS = sorted(df[config.ID].unique())

        features = []
        for i in IDS:
            feature = {}
            feature[self.config.ID] = i
            game = df.query(f"{config.ID} == {i} and time <= {config.CUTOFF}")
            for col in config.NUMERIC_FEATURES_TEMPORAL:
                val = game[[col]].values
                feature[f'{col}_mean'] = val.mean()
                feature[f'{col}_std'] = val.std()
                feature[f'{col}_min'] = val.min()
                feature[f'{col}_max'] = val.max()
                feature[f'{col}_last_1'] = val.flatten()[-1]
                feature[f'{col}_last_2'] = val.flatten()[-2]
                feature[f'{col}_last_3'] = val.flatten()[-3]
            for col in config.NUMERIC_FEATURES_CONSTANT:
                feature[col] = game[col].mean()
            feature['intercept'] = 1
            features.append(feature)

        features = pd.DataFrame(features)
        return features
 
class WeatherDataset(Dataset):
    def __init__(self, **kwargs):
        self.config = C.WeatherConfig
        super().__init__(**kwargs)

    def create_features(self, df):
        logging.info("Generating features for training dataset...")
        config = self.config
        IDS = sorted(df[config.ID].unique())

        df = df.query("time == 0").reset_index()
        onehot_encoder = OneHotEncoder(drop="if_binary")
        df_onehot = pd.DataFrame(onehot_encoder.fit_transform(
            df[config.CATEGORICAL_FEATURES]).toarray())
        df_onehot.columns = onehot_encoder.get_feature_names(config.CATEGORICAL_FEATURES)
        columns = [f"onehot_feature{k}" for k in df_onehot.columns]
        config.ALL_FEATURES = df_onehot.columns.tolist() + config.NUMERIC_FEATURES

        features = df_onehot.join(df[config.NUMERIC_FEATURES])

        features = features.join(df[config.ID])
        features = features.sort_values(config.ID)
        return features 
