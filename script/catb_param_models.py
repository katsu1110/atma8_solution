import numpy as np
import pandas as pd
from sklearn import utils
from catboost import CatBoostRegressor, CatBoostClassifier

def catb_model(cls, train_set, val_set):
    """
    CatBoost hyperparameters and models
    """

    # verbose
    verbosity = 100 if cls.verbose else 0

    # list is here: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    params = { 'task_type': "CPU",
                'learning_rate': cls.cfg.catboost.learning_rate, 
                'iterations': cls.cfg.catboost.iterations,
                'colsample_bylevel': cls.cfg.catboost.colsample_bylevel,
                'random_seed': cls.seed,
                'use_best_model': True,
                'early_stopping_rounds': cls.cfg.catboost.early_stopping_rounds
                }
    if cls.task == "regression":
        params["loss_function"] = "RMSE"
        params["eval_metric"] = "RMSE"
    elif cls.task == "binary":
        params["loss_function"] = "Logloss"
        params["eval_metric"] = "AUC"
    elif cls.task == "multiclass":
        params["loss_function"] = "MultiClass"
        params["eval_metric"] = "MultiClass"

    # modeling
    if cls.task == "regression":
        model = CatBoostRegressor(**params)
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        cw = utils.class_weight.compute_class_weight('balanced', np.unique(train_set['y']), train_set['y'])
        params['class_weights'] = cw
        model = CatBoostClassifier(**params)
    model.fit(train_set['X'], train_set['y'], eval_set=(val_set['X'], val_set['y']),
        verbose=verbosity, cat_features=cls.categoricals)
    
    # feature importance
    fi = model.get_feature_importance()

    return model, fi