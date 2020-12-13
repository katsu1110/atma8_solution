import numpy as np
import pandas as pd
import os
import sys
import gc
import pickle
import json
import tempfile
import pathlib
import datetime
from loguru import logger
from sklearn import utils
from catboost import CatBoostRegressor, CatBoostClassifier
from omegaconf import DictConfig
import hydra

from preprocess import preprocess
from utils import validation_score, to_pickle, from_pickle, reduce_mem_usage
from run_models import RunModel

def load_data(cfg):
    train = pd.read_csv(pathlib.Path(cfg.path + 'input/train.csv'))
    test = pd.read_csv(pathlib.Path(cfg.path + 'input/test.csv'))
    return train, test

@hydra.main(config_name='../config/config.yml')
def main(cfg):
    # --------------------
    # load data
    # --------------------
    logger.debug('load data...')
    train, test = load_data(cfg)

    # make folders if not exist
    os.makedirs(pathlib.Path(cfg.path + f'output/feature_importance/', exist_ok=True))
    os.makedirs(pathlib.Path(cfg.path + f'output/oof/', exist_ok=True))
    os.makedirs(pathlib.Path(cfg.path + f'output/submission/', exist_ok=True))
    os.makedirs(pathlib.Path(cfg.path + f'output/feature_engineered/', exist_ok=True))

    # --------------------
    # preprocess
    # --------------------
    logger.debug('preprocess...')
    EXPERIMENT_NAME = '{}_{}'.format(datetime.datetime.now().strftime('%Y%m%d'), cfg.experiment)
    train, test, features, cat_feats, target, group = preprocess(train, test)
    logger.debug(f'{len(features)} all features')

    # save feature engineered data
    assert 'JP_Sales' in train.columns.values.tolist()
    assert 'data_id' in test.columns.values.tolist()
    to_pickle(pathlib.Path(cfg.path + f'output/feature_engineered/train{EXPERIMENT_NAME}.pkl'), train)
    to_pickle(pathlib.Path(cfg.path + f'output/feature_engineered/test{EXPERIMENT_NAME}.pkl'), test)
    logger.debug('Preprocess & feature engineering done!')

# main run
if __name__ == '__main__':
    main()