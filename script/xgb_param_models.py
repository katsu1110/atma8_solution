import numpy as np
import pandas as pd
from sklearn import utils
import xgboost as xgb
import operator

def xgb_model(cls, train_set, val_set):
    """
    XGB hyperparameters and models
    """

    # verbose
    verbosity = 100 if cls.verbose else 0

    # list is here: https://xgboost.readthedocs.io/en/latest/parameter.html
    params = {
        'colsample_bytree': cls.cfg.xgboost.colsample_bytree,                 
        'learning_rate': cls.cfg.xgboost.learning_rate,
        'max_depth': cls.cfg.xgboost.max_depth,
        'subsample': cls.cfg.xgboost.subsample,
        'min_child_weight': cls.cfg.xgboost.min_child_weight,
        'gamma': cls.cfg.xgboost.gamma,
        'alpha': cls.cfg.xgboost.alpha,
        'lambda': cls.cfg.xgboost.lambda_val,
        'seed': cls.seed,
        'n_estimators': cls.cfg.xgboost.n_estimators
    }
    if cls.task == "regression":
        params["objective"] = 'reg:squarederror'
        params["eval_metric"] = "rmse"
    elif cls.task == "binary":
        params["objective"] = 'binary:logistic'
        params["eval_metric"] = "auc"
        # cw = utils.class_weight.compute_class_weight('balanced', np.unique(train_set['y']), train_set['y'])
        # params['scale_pos_weight'] = int(cw[1] / cw[0])
    elif cls.task == "multiclass":
        params["objective"] = 'multi:softmax'
        params["eval_metric"] = "mlogloss" 

    # modeling
    if cls.task == "regression":
        model = xgb.XGBRegressor(**params)
    elif (cls.task == "binary") | (cls.task == "multiclass"):
        model = xgb.XGBClassifier(**params)
    model.fit(train_set['X'], train_set['y'], eval_set=[(val_set['X'], val_set['y'])],
                    early_stopping_rounds=80, verbose=verbosity)

    # feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    fi = np.zeros(len(cls.features))
    for i, f in enumerate(cls.features):
        try:
            fi[i] = df.loc[df['feature'] == f, "fscore"].iloc[0]
        except: # ignored by XGB
            continue

    return model, fi