import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import glob
import pyreadr
from torch import nn
import pandas as pd
import fire
import sys
from tqdm import tqdm
from pytorch_lightning import Trainer
import logging

from constants import *

from .seq2seq import MQSeq2Seq
from .dataset import *
from .util import *
from .quantile import QuantileLoss, CoverageLoss



class Model(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        device = "cuda" if self.hparams['gpus'] is not None else "cpu"
        self.env = self.hparams['env']
        self.yhat = {}
        if self.env == "NBA":
            self.config = NBAConfig
            self.input_dim = self.output_dim = 6
            self.step = 24
            self.start = 25
            self.end = 49
            self.ID = "GAME_ID"
        else:
            self.config = WeatherConfig
            self.input_dim = self.output_dim = 70 
            self.step = 7
            self.start = 1
            self.end = 8
            self.ID = "obs"
        self.quantiles = np.linspace(0.01, 0.99, self.hparams['n_quantile'])
        self.model = MQSeq2Seq(self.input_dim, 
                               self.hparams['n_quantile'],
                               n_layers=self.hparams['n_layers'], 
                               quantiles=self.quantiles, 
                               dropout=self.hparams['dropout'],
                               device=device).float()
        if self.hparams['loss_fn'] == "quantile":
            self.loss = QuantileLoss(self.quantiles)
        else:
            self.loss = CoverageLoss(self.quantiles)
        self.batch_size = self.hparams['batch_size']
        self.learning_rate = self.hparams['learning_rate']
       

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y = y.float()
        yhat = self.model(x, self.step)
        loss = self.loss(yhat, y)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y = y.float()
        yhat = self.model(x, self.step)
        loss = self.loss(yhat, y)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_val_loss', avg_loss)

    def test_step(self, batch, batch_nb):
        x, y = batch
        x = x.float()
        y = y.float()
        self.model.train()
        with torch.no_grad():
            yhat = self.model(x.repeat(self.hparams['sample'], 1, 1), self.step, sampling=True)
        self.model.eval()
        self.yhat[batch_nb] = yhat
        
        return {'yhat': yhat,
                }

    def test_epoch_end(self, outputs):
        yhat = torch.stack([self.yhat[i] for i in range(max(self.yhat.keys()))])
        N, M, T = yhat.shape
        IDS = self.test_dataset.data[self.ID].unique()
        paths = yhat.cpu().detach().numpy()
        results = []
        for n in tqdm(range(N)):
            result = pd.DataFrame(paths[n], columns=list(range(self.start, self.end)))
            result = result.stack().reset_index()
            result.columns = ['sample_id', 'time', 'pred_outcome']
            result["obs_id"] = IDS[n]
            results.append(result)
        results = pd.concat(results)
        final_rows = results.time==self.config.T
        results.loc[final_rows, "pred_outcome"] = 1.0 * (results.loc[final_rows, "pred_outcome"] >0.5) 
        #paths.to_csv(self.config.SAMPLE_MQLSTM_PATH, index=False)
        pyreadr.write_rds(self.config.SAMPLE_MQLSTM_PATH, results)

        logging.info(f"Sampled path saved to [{self.config.SAMPLE_MQLSTM_PATH}]")

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate)]

    def train_dataloader(self):
        if self.env == "NBA":
            dataset = NBADataset
        else:
            dataset = WeatherDataset
        self.train_dataset = dataset(split="train")
        return DataLoader(self.train_dataset, shuffle=True,
                          batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        if self.env == "NBA":
            dataset = NBADataset
        else:
            dataset = WeatherDataset
        self.val_dataset = dataset(split="test")
        return DataLoader(self.val_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        if self.env == "NBA":
            dataset = NBADataset
        else:
            dataset = WeatherDataset
        self.test_dataset = dataset(split="test")
        return DataLoader(self.test_dataset, shuffle=False,
                          batch_size=1, num_workers=8)


def mqlstm(save_dir=SANDBOX_DIR,
          exp_name="mqlstm",
          env="NBA",
          gpus=None,
          accelerator="dp",
          gradient_clip_val=2,
          max_epochs=100,
          patience=5,
          sample=100,
          limit_train_batches=1.0,
          limit_val_batches=1.0,
          limit_test_batches=1.0,
          weights_summary=None,
          batch_size=128,
          dropout=0.00,
          loss_fn="quantile",
          learning_rate=1e-4,
          n_quantile=200,
          n_layers=4,
          ):
    exp_name = f"{exp_name}_{env}"
    args = Args(locals())

    save_path = os.path.join(save_dir, exp_name)

    init_exp_folder(args)
    m = Model(args)
    trainer = Trainer(accelerator=accelerator,
                      limit_train_batches=limit_train_batches,
                      limit_val_batches=limit_val_batches,
                      limit_test_batches=limit_test_batches,
                      gpus=gpus,
                      logger=get_logger(save_dir, exp_name),
                      callbacks=[get_ckpt_callback(save_dir, exp_name),
                                 get_early_stop_callback(patience)],
                      weights_save_path=save_path,
                      gradient_clip_val=gradient_clip_val,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(m)
    ckpt_path = list(glob.glob(os.path.join(save_path, "*.ckpt")))[0]
    m = Model.load_from_checkpoint(ckpt_path)
    trainer.test(m)

if __name__ == "__main__":
    fire.Fire()
