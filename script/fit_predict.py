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
import itertools
from loguru import logger
from sklearn import utils
from sklearn import model_selection
from catboost import CatBoostRegressor, CatBoostClassifier
import mlflow
from omegaconf import DictConfig
import hydra

from preprocess import preprocess
from utils import validation_score, to_pickle, from_pickle, reduce_mem_usage
from run_models import RunModel

# ---------------------------
# utils
# ---------------------------
def create_folds(data, target_col):
    num_bins = np.int(1 + np.log2(len(data)))
    bins = pd.cut(
        data[target_col],
        bins=num_bins,
        labels=False
    )
    return bins.values

def fit_single_model(train, test, target, features, cat_feats, group,
        model_name='lgb', task='regression', cv='GroupKFold', n_splits=5, nsa=3, cfg={}):
    oof = np.zeros(train.shape[0])
    y_pred = np.zeros(test.shape[0])
    for s in range(nsa): # seed average
        # model fitting
        model = RunModel(train, test, target, features, categoricals=cat_feats,
                group=group, model=model_name, task=task, cv_method=cv, n_splits=n_splits,
                cfg=cfg, target_encoding=False, seed=cfg.seed+s**2, scaler=None)

        # average results
        oof += model.oof / nsa
        y_pred += model.y_pred / nsa

    # feature importance
    fi_df = model.fi_df.drop_duplicates(subset=['features', 'importance_mean'])
    fi_df = fi_df.sort_values(by='importance_mean', ascending=False)
    
    return oof, y_pred, fi_df[['features', 'importance_mean']], model

def after_modeling(y_true, y_pred, fi_df, m, cv, cfg, EXPERIMENT_NAME):
    # feature importance
    logger.debug(f'Top 10 features for {m} by {cv}:')
    logger.debug(fi_df['features'].values[:10].tolist())
    savepath = pathlib.Path(cfg.path + f'output/feature_importance/feature_importance_{m}_{cv}_{EXPERIMENT_NAME}.csv')
    fi_df.to_csv(savepath, index=False)
    
    # evaluate results
    logger.debug(f'computing validation score for {m} by {cv}...')
    score = validation_score(y_true, y_pred)
    logger.debug('Overall score for {} by {} = {}'.format(m, cv, score))
    return savepath, score

@hydra.main(config_name='../config/config.yml')
def main(cfg):
    # --------------------
    # setup
    # --------------------
    logger.debug('setup...')
    EXPERIMENT_NAME = '{}_{}'.format(datetime.datetime.now().strftime('%Y%m%d'), cfg.experiment)
    logger.add(pathlib.Path(cfg.path + f'output/log/{EXPERIMENT_NAME}.log'), enqueue=True, backtrace=True)
    scores = {}

    # --------------------
    # load data
    # --------------------
    logger.debug('load preprocessed & fe data...')
    train = from_pickle(pathlib.Path(cfg.path + f'output/feature_engineered/train{EXPERIMENT_NAME}.pkl'))
    test = from_pickle(pathlib.Path(cfg.path + f'output/feature_engineered/test{EXPERIMENT_NAME}.pkl'))
    
    # new targets
    ids = 'data_id'
    target = 'Global_Sales'
    non_targets = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    for c in list(itertools.combinations(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], 3)):
        new_tg = c[0] + '_' + c[1] + '_' + c[2]
        train[new_tg] = train[c[0]] + train[c[1]] + train[c[2]]
        non_targets.append(new_tg)
    
    # log transform
    for t in non_targets + [target]:
        train[t] = np.log1p(train[t])
        assert t in train.columns.values.tolist()
        logger.debug(t + ' exists')

    # features to use
    drops = [
        'Platform_Genre_nunique_by_Publisher',
        'Year_of_Release_nunique_by_Publisher',
        'Platform_nunique_by_Publisher',
        'Platform_nunique_by_Developer',
        'Genre_nunique_by_Developer',
        'category2vec_0_max',
        'category2vec_0_min'
        ]
    drops += [f for f in test.columns.values.tolist() if ('_x_' in f) & f.endswith('Publisher')]
    if cfg.fit.cv == 'KFold':
        group = 'Publisher'
    elif cfg.fit.cv == 'StratifiedKFold2':
        train['bins'] = create_folds(train, target)
        group = 'bins'
    cat_feats = ['Platform', 'Genre', 'Rating', 'Platform_Genre']
    drops += [ids, group, 'Publisher', target] + non_targets
    features = [f for f in test.columns.values.tolist() if f not in drops]
    cat_feats = [f for f in cat_feats if f in features]

    # --------------------
    # 1st feature selection (adversarial validation)
    # --------------------
    m = 'lgb'
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], ignore_index=True)
    auc = 1
    counts = 0
    while auc > cfg.fit.adversarial_validation:
        if counts > 0:
            drops = fi_df['features'].values[:int(0.02*len(features))].tolist()
            print('drops:', drops)
            features = fi_df['features'].values[int(0.02*len(features)):].tolist()
            cat_feats = [f for f in cat_feats if f in features]
        
        # fit
        oof_, y_pred_, fi_df, model = fit_single_model(df, df, 'is_train',
            features, cat_feats, group, m, task='binary', cv='StratifiedKFold',
            n_splits=2, nsa=1, cfg=cfg)
        auc = model.score
        counts += 1
        logger.debug('adversarial validation score (auc) = {}'.format(auc))

    if cfg.fit.adversarial_validation < 1:
        savepath_adv = pathlib.Path(cfg.path + f'output/feature_importance/adv_{EXPERIMENT_NAME}.csv')
        fi_df.to_csv(savepath_adv, index=False)
    else:
        logger.debug('skipping adversarial validation...')

    # --------------------
    # 2nd feature selection (simply fit and select top 64% features)
    # --------------------
    # initialize for stacking
    m = 'lgb'
    n = 'full_feats_'+target
    cv = cfg.fit.cv
    task = 'regression'
    oof_df = pd.DataFrame()
    oof_df[target] = train[target].values
    oof_df[group] = train[group].values
    oof_df[ids] = train[ids].values
    ypred_df = pd.DataFrame()
    ypred_df[ids] = test[ids].values
    assert oof_df[ids].values[-1] + 1 == ypred_df[ids].values[0]

    # fit
    oof_, y_pred_, fi_df, _ = fit_single_model(train, test, target, features, cat_feats, group,
        model_name=m, task=task, cv=cv, n_splits=cfg.fit.nfold, nsa=cfg.fit.nsa, cfg=cfg)
    savepath, score = after_modeling(train[target].values, oof_, fi_df, m, cv, cfg, EXPERIMENT_NAME)
    scores[f'lgb_full{len(features)}_features'] = score

    # feature selection
    features = fi_df['features'].values[:int(0.64 * len(features))].tolist()
    cat_feats = [f for f in cat_feats if f in features]
    logger.debug(f'{len(features)} selected features')

    # assign
    oof_df[n] = oof_
    ypred_df[n] = y_pred_

    # ------------------------
    # fit for non-targets (for stacking)
    # ------------------------
    non_targets = [f for f in non_targets if f not in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    for i, t in enumerate(non_targets):
        # fit
        if (cfg.mode == 'debug') & (t != target):
            continue
        n = f'pred_{t}'
        logger.debug('# -----------------------')
        logger.debug(f'predicting {t}...')
        logger.debug('# -----------------------')
        oof_, y_pred_, fi_df, _ = fit_single_model(train, test, t,
            features, cat_feats, group, model_name=m, task=task, cv=cv,
            n_splits=cfg.fit.nfold, nsa=cfg.fit.nsa, cfg=cfg)
        
        # assign
        oof_df[n] = oof_
        ypred_df[n] = y_pred_

    # ------------------------
    # fit for target
    # ------------------------
    models = ['catb', 'xgb']
    for m in models:
        # fitting
        logger.debug('# -----------------------')
        logger.debug(f'predicting {target} by {m}...')
        logger.debug('# -----------------------')
        oof_, y_pred_, fi_df, _ = fit_single_model(train, test, target,
            features, cat_feats, group, model_name=m, task=task, cv=cv,
            n_splits=cfg.fit.nfold, nsa=cfg.fit.nsa, cfg=cfg)

        # assign
        oof_df[f'{m}_{target}'] = oof_
        ypred_df[f'{m}_{target}'] = y_pred_

    # store validation scores
    for n in ypred_df.columns.values.tolist():
        if target in n:
            score = validation_score(train[target].values, oof_df[n].values)
            logger.debug('Overall score for {} = {}'.format(n, score))
            scores[n] = score

    # -------------------
    # stacking
    # -------------------
    logger.debug('stacking ensemble...')
    m = 'linear'
    oof = np.zeros(train.shape[0])
    y_pred = np.zeros(test.shape[0])
    logger.debug(f'fitting {m}...')
    stacking_feats = [f for f in ypred_df.columns.values.tolist() if f not in drops]
    oof, y_pred, fi_df, _ = fit_single_model(oof_df, ypred_df, target,
        stacking_feats, [], group, model_name=m, task=task, cv=cv,
            n_splits=cfg.fit.nfold, nsa=cfg.fit.nsa, cfg=cfg)

    # -------------------
    # evaluate results
    # -------------------
    ss = pd.read_csv(pathlib.Path(cfg.path + 'input/atmacup8_sample-submission.csv'))
    ss[target] = np.expm1(y_pred)
    score = validation_score(train[target].values, oof)
    logger.debug('Overall score for ensemble = {}'.format(score))
    scores['final'] = score

    # --------------------
    # save files
    # --------------------
    logger.debug('saving files...')

    # oof, submissions
    fi_df.to_csv(pathlib.Path(cfg.path + f'output/feature_importance/weights_{EXPERIMENT_NAME}.csv'), index=False)
    np.save(pathlib.Path(cfg.path + f'output/oof/oof_final{EXPERIMENT_NAME}'), oof)
    ss.to_csv(pathlib.Path(cfg.path + f'output/submission/{EXPERIMENT_NAME}.csv'), index=False)
    
    # --------------------
    # mlflow
    # --------------------
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with tempfile.TemporaryDirectory() as tmp_dir, mlflow.start_run() as run:
        # Added this line
        logger.debug('tracking uri:', mlflow.get_tracking_uri())
        logger.debug('artifact uri:', mlflow.get_artifact_uri())

        # hyperparameters
        mlflow.log_params(cfg.lightgbm)
        if cfg.mode != 'debug':
            mlflow.log_params(cfg.catboost)
            mlflow.log_params(cfg.xgboost)
        
        # settings
        mlflow.log_param('mode', cfg.mode)
        mlflow.log_param('seed', cfg.seed)
        mlflow.log_param('ensemble', cfg.ensemble)

        # scores
        for k, v in scores.items():
            mlflow.log_metric(k, v)

        # outputs
        artifacts = {
            'ensemble_weights': cfg.path + f'output/feature_importance/weights_{EXPERIMENT_NAME}.csv',
            'feature_importance': savepath,
            'oof': cfg.path + f'output/oof/oof_final{EXPERIMENT_NAME}.npy',
            'submission': cfg.path + f'output/submission/{EXPERIMENT_NAME}.csv',
        }
        for name, file_path in artifacts.items():
            mlflow.log_artifact(pathlib.Path(file_path))
    
    logger.debug('all done')

# main run
if __name__ == '__main__':
    main()