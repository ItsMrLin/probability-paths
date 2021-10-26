from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pyreadr
import pandas as pd
import numpy as np
import fire
from tqdm import tqdm
import statsmodels.api as sm
from scipy.special import logit, expit
import logging

import constants as C

from .dataset import NBADataset, WeatherDataset

def get_start(df, config):
    t = config.CUTOFF
    label_cur = df.query(f"time=={t}")[config.PREDICTION]
    label_logit = logit(np.clip(label_cur, EPSILON, 1-EPSILON)) 
    return label_logit

def lr(M=100, 
       env="nba",
       n_lag=2):
    if env == "nba":
        config = C.NBAConfig
        dataset = NBADataset()
    else:
        config = C.WeatherConfig
        dataset = WeatherDataset()
    models = {}
    rhats = {}
    features = dataset.get_features("train")
    # Training
    logging.info("Start fitting...")
    for t in tqdm(range(config.CUTOFF + 1, config.T + 1)):
        X = features.copy()
        for lag in range(1, n_lag+1):
            t_lag = t - lag
            if t_lag >= 0:
                X[f'lag_{lag}'] = dataset.get_label(t_lag, "train")

        y = dataset.get_label(t, "train")
        m = Pipeline([('normalize', StandardScaler()),
                      ('lr', LinearRegression())])
        index_finite = np.isfinite(y)
        X = X[index_finite]
        y = y[index_finite]
        m.fit(X, y)
        rhat = (y - m.predict(X)) ** 2
        #m = sm.OLS(y.values, X).fit()
        models[t] = m
        rhats[t] = np.sqrt(rhat.mean())

    # Sampling
    features = dataset.get_features("test")
    IDS = dataset.get_ids("test")
    logging.info("Start sampling...")
    paths = []
    for m in tqdm(range(M)):
        preds = {} 
        for t in range(config.CUTOFF + 1, config.T + 1):
            X = features.copy() 
            for lag in range(1, n_lag+1):
                t_lag = t - lag
                if t_lag >= 0:
                    if t_lag in preds:
                        X[f'lag_{lag}'] = preds[t_lag]
                    else: 
                        X[f'lag_{lag}'] = dataset.get_label(t_lag, "test")
            
            pred = models[t].predict(X)
            pred += np.random.randn(pred.shape[0]) * rhats[t]
            preds[t] = pred

            path = expit(pred)
            paths.append(pd.DataFrame({"pred_outcome": path,
                                       "obs_id": IDS,
                                       "time": t,
                                       "sample_id": m+1})) 
    paths = pd.concat(paths)
    final_rows = paths.time==config.T
    paths.loc[final_rows, "pred_outcome"] = 1.0 * (paths.loc[final_rows, "pred_outcome"] >0.5) 
    paths['obs_id'] = paths['obs_id'].astype("int") 
    pyreadr.write_rds(config.SAMPLE_LR_PATH, paths)
    logging.info(f"Sampled path saved to [{config.SAMPLE_LR_PATH}]")


if __name__ == "__main__":
    fire.Fire(main)
