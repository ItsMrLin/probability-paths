import json
import os
import shutil
from os.path import join
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


LIGHTNING_CKPT_PATH = 'lightning_logs/version_0/checkpoints/'
LIGHTNING_TB_PATH = 'lightning_logs/version_0/'
LIGHTNING_METRICS_PATH = 'lightning_logs/version_0/metrics.csv'


def get_ckpt_dir(save_path, exp_name):
    return os.path.join(save_path, exp_name)


def get_ckpt_callback(save_path, exp_name):
    ckpt_dir = os.path.join(save_path, exp_name)
    return ModelCheckpoint(dirpath=ckpt_dir,
                           filename="ckpt",
                           save_top_k=1,
                           verbose=True,
                           monitor='val_loss',
                           mode='min',
                           )


def get_early_stop_callback(patience=10):
    return EarlyStopping(monitor='val_loss',
                         patience=patience,
                         verbose=True,
                         mode='min')


def get_logger(save_path, exp_name):
    exp_dir = os.path.join(save_path, exp_name)
    return TestTubeLogger(save_dir=exp_dir,
                          name='lightning_logs',
                          version="0")


class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(args[0])

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            AttributeError("No such attribute: " + name)


def init_exp_folder(args):
    save_dir = os.path.abspath(args.get("save_dir"))
    exp_name = args.get("exp_name")
    exp_path = join(save_dir, exp_name)
    exp_metrics_path = join(exp_path, "metrics.csv")
    exp_tb_path = join(exp_path, "tb")

    # init exp path
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)
        #raise FileExistsError(f"Experiment path [{exp_path}] already exists!")
    os.makedirs(exp_path, exist_ok=True)

    # dump hyper-parameters/arguments
    json.dump(locals(),
              open(join(save_dir, exp_name, "args.json"), "w+"))

