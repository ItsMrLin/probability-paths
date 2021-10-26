import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from constants import *


class NBADataset(torch.utils.data.Dataset):
    def __init__(self, t=24, split="train"):
        self.T = 48
        self.t = t
        if split == "train":
            self.data = pd.read_csv(NBAConfig.TRAIN_PATH)
        else:
            self.data = pd.read_csv(NBAConfig.TEST_PATH)
        self.ID = self.data.GAME_ID.unique().tolist()
        self.features = [
            'home_score',
            "time",
            "score_margin",
            "home_win_rate",
            "visitor_win_rate",
            "pred_outcome"]
        self.targets = "pred_outcome"

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, index):
        ID = self.ID[index]
        game = self.data.query(f"GAME_ID == {ID}").reset_index()
        X = game.loc[:self.t, self.features].values.astype(np.float32)
        Y = game.loc[self.t + 1:, self.targets].values.astype(np.float32)
        return X, Y
    
    
class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, t=0, split="train"):
        self.config = WeatherConfig
        self.T = self.config.T
        self.t = t
        if split == "train":
            self.data = pd.read_csv(WeatherConfig.TRAIN_PATH)
        else:
            self.data = pd.read_csv(WeatherConfig.TEST_PATH)
            
        self.ID = self.data.obs.unique().tolist()
        
        onehot_encoder = OneHotEncoder()
        df_onehot = pd.DataFrame(onehot_encoder.fit_transform(
            self.data[self.config.CATEGORICAL_FEATURES]).toarray())
        columns = [f"onehot_feature{k}" for k in df_onehot.columns]
        df_onehot.columns = columns
        self.config.ALL_FEATURES = df_onehot.columns.tolist() + self.config.NUMERIC_FEATURES
        self.data = df_onehot.join(self.data[self.config.NUMERIC_FEATURES + ["obs", "pred_outcome"]])
        
        self.features = columns + ['pred_outcome']
        self.targets = "pred_outcome"

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, index):
        ID = self.ID[index]
        game = self.data.query(f"obs == {ID}").reset_index()
        X = game.loc[:self.t, self.features].values.astype(np.float32)
        Y = game.loc[self.t + 1:, self.targets].values.astype(np.float32)
        return X, Y
