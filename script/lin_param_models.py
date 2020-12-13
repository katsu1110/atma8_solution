import numpy as np
from sklearn import linear_model

def lin_model(cls, train_set, val_set):
    """
    Linear model hyperparameters and models
    """
    if cls.task == "regression":
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        params = {
                'alpha': cls.cfg.linear.alpha, 
                'solver': 'lsqr', 
                'fit_intercept': True,
                'max_iter': cls.cfg.linear.max_iter, 
                'tol': cls.cfg.linear.tol,
                'random_state': cls.seed,
        }
    else:
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        params = {
                "C": cls.cfg.linear.alpha, 
                "solver": "lbfgs", 
                "warm_start": False,
                "max_iter": cls.cfg.linear.max_iter,
                "fit_intercept": True,
                "random_state": cls.seed,
                "tol": cls.cfg.linear.tol,
                "verbose": 1, 
                "class_weight": "balanced", 
        }
        if cls.task == "multiclass":
            params["multi_class"] = "multinomial"
    
    if cls.task == "regression":      
        model = linear_model.Ridge(**params)
    else:
        model = linear_model.LogisticRegression(**params)

    model.fit(train_set['X'], train_set['y'])

    # feature importance (for multitask, absolute value is computed for each feature)
    if cls.task == "multiclass":
        fi = np.mean(np.abs(model.coef_), axis=0).ravel()
    else:
        fi = model.coef_.ravel()

    return model, fi
