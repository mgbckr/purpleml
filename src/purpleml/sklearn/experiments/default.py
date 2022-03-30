import collections
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
import sklearn.pipeline
import numpy as np
from purpleml.sklearn.experiments.utils import Experiment

# TODO:
# * regression
# * skopt (in addition to random)
# * make cv more configurable
# * add inner group support? maybe not


def parse_preprocessing(preprocessing):
    if preprocessing == "standard":
        preprocessing = [sklearn.preprocessing.StandardScaler()]
    if preprocessing == "impute and scale":
        preprocessing = [
            sklearn.impute.SimpleImputer(strategy="median"),
            sklearn.preprocessing.StandardScaler()]
    elif preprocessing is None:
        preprocessing = []
    return preprocessing


def init_default_experiments_classification(
        preprocessing="standard",
        prefix="model_",
        return_dict=False):
    """
    Notes:
    * We set `tol` and `max_iter` for `LogisticRegression` to `1E-4` and `1E3`, respectively,
      in order to match default regression equivalents..

    Parameters
    ----------
    preprocessing
    prefix
    return_dict

    Returns
    -------

    """

    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # default

    experiments[f"{prefix}rf-100"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.ensemble.RandomForestClassifier(n_estimators=100)))

    experiments[f"{prefix}lr-default"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LogisticRegression()))

    experiments[f"{prefix}lr-ridge"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LogisticRegression(
                solver="lbfgs",
                tol=1E-4,  # default
                max_iter=1E3  # increased from 100 to 1000
            )))

    experiments[f"{prefix}lr-lasso"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LogisticRegression(
                penalty="l1",
                solver="saga",
                tol=1E-4,  # default
                max_iter=1E3  # increased from 100 to 1000
            )))

    experiments[f"{prefix}lr-en-fast"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                solver="saga",
                tol=1E-4,  # default
                max_iter=1E3  # increased from 100 to 1000
            )))

    experiments[f"{prefix}svm-rbf"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.svm.SVC()))

    experiments[f"{prefix}svm-linear"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.svm.SVC(kernel="linear")))

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments


def init_default_experiments_regression(
        preprocessing="standard",
        prefix="model_",
        return_dict=False):

    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # default

    experiments[f"{prefix}rf-100"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.ensemble.RandomForestRegressor(n_estimators=100)))

    experiments[f"{prefix}lr-default"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LinearRegression()))

    experiments[f"{prefix}lr-ridge"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.Ridge()))

    experiments[f"{prefix}lr-lasso"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.Lasso()))

    experiments[f"{prefix}lr-en-fast"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.ElasticNet()))

    experiments[f"{prefix}svm-rbf"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.svm.SVR()))

    experiments[f"{prefix}svm-linear"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.svm.SVR(kernel="linear")))

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments


def init_default_experiments_classification_optimized_linear(
        preprocessing="standard",
        prefix="model_",
        exclude_limited_elastic_net=True,
        exclude_full_elastic_net=False,
        return_dict=False):
    """
    Notes:
    * We set `tol` and `max_iter` for `LogisticRegression` to `1E-4` and `1E3`, respectively,
      in order to match default regression equivalents.
    * Groups for inner cross validation not supported.

    Parameters
    ----------
    preprocessing
    prefix
    exclude_limited_elastic_net
    exclude_full_elastic_net
    return_dict

    Returns
    -------

    """

    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # default

    experiments[f"{prefix}lr-ridge_cv"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LogisticRegressionCV(
                solver="lbfgs",
                tol=1E-4,
                max_iter=1E3
            )))

    experiments[f"{prefix}lr-lasso_cv"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                tol=1E-4,
                max_iter=1E3
            )))

    if not exclude_limited_elastic_net:
        experiments[f"{prefix}lr-en-fast_cv_l1-ratio_0.5"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.linear_model.LogisticRegressionCV(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratios=[0.5],
                    tol=1E-4,
                    max_iter=1E3
                )))

    if not exclude_full_elastic_net:
        experiments[f"{prefix}lr-en-fast_cv"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.linear_model.LogisticRegressionCV(
                    penalty="elasticnet",
                    solver="saga",
                    # see docs for ElasticNetCV for choice of l1_ratios
                    l1_ratios=np.concatenate([np.linspace(0.1, 1, 10), [0.95, 0.98, 0.99]]),
                    tol=1E-4,
                    max_iter=1E3
                )))

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments


def init_default_experiments_classification_optimized_crosssplit_linear(
        preprocessing="standard",
        cross_split_scoring=None,
        cv=None,
        prefix="model_",
        exclude_limited_elastic_net=True,
        exclude_full_elastic_net=True,
        groups=True,
        return_dict=False):
    """
    Parameters
    ----------
    preprocessing
    prefix
    exclude_limited_elastic_net
    exclude_full_elastic_net
    return_dict

    Returns
    -------

    """

    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # default

    import nalabtools.sklearn.model_selection
    import sklearn.metrics

    if cross_split_scoring is None:
        cross_split_scoring = dict(score_func=sklearn.metrics.roc_auc_score, prediction="predict_proba_1")
        cross_split_scoring_svm = dict(score_func=sklearn.metrics.roc_auc_score, prediction="decision_function")

    if cv is None:
        if groups:
            cv = sklearn.model_selection.LeaveOneGroupOut()
        else:
            cv = sklearn.model_selection.LeaveOneOut()

    # lasso
    e = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                sklearn.linear_model.LogisticRegression(penalty="l1", solver="liblinear"),
                param_grid=dict(
                    C=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                ),
                cross_split_scoring=cross_split_scoring,
                cv=cv
            )
        )
    )
    if groups:
        e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
    experiments[f"{prefix}lr-lasso_cv-crosssplit"] = e

    # ridge
    e = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                sklearn.linear_model.LogisticRegression(penalty="l2"),
                param_grid=dict(
                    C=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                ),
                cross_split_scoring=cross_split_scoring,
                cv=cv
            )
        )
    )
    if groups:
        e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
    experiments[f"{prefix}lr-ridge_cv-crosssplit"] = e

    # linear svm

    e = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                sklearn.svm.SVC(kernel="linear"),
                param_grid=dict(
                    # https://stackoverflow.com/questions/26337403/what-is-a-good-range-of-values-for-the-svm-svc-hyperparameters-to-be-explored
                    C=np.logspace(-3, 2, 6),
                ),
                cross_split_scoring=cross_split_scoring_svm,
                cv=cv
            )
        )
    )
    if groups:
        e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
    experiments[f"{prefix}svm-linear_cv-crosssplit"] = e

    # elastic net

    if not exclude_limited_elastic_net:
        e = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                    sklearn.linear_model.LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        tol=1E-4,
                        max_iter=1E3  # increase by *10
                    ),
                    param_grid=dict(
                        C=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                        l1_ratio=[0.5],
                    ),
                    cross_split_scoring=cross_split_scoring,
                    cv=cv
                )
            )
        )
        if groups:
            e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
        experiments[f"{prefix}lr-en_cv-crosssplit_l1-ratio_0.5"] = e

    if not exclude_full_elastic_net:
        e = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                    sklearn.linear_model.LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        tol=1E-4,
                        max_iter=1E3  # increase by *10
                    ),
                    param_grid=dict(
                        C=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                        l1_ratio=np.concatenate([np.linspace(0.1, 1, 10), [0.95, 0.98, 0.99]]), # only for elastic net
                    ),
                    cross_split_scoring=cross_split_scoring,
                    cv=cv
                )
            )
        )
        if groups:
            e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
        experiments[f"{prefix}lr-en_cv-crosssplit"] = e

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments


def init_default_experiments_regression_optimized_linear(
        preprocessing="standard",
        prefix="model_",
        exclude_limited_elastic_net=True,
        exclude_full_elastic_net=False,
        return_dict=False):
    """
    Notes
    * Groups for inner cross validation not supported.

    Parameters
    ----------
    preprocessing
    prefix
    exclude_limited_elastic_net
    exclude_full_elastic_net
    return_dict

    Returns
    -------

    """

    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # default

    experiments[f"{prefix}lr-ridge_cv"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.RidgeCV()))

    experiments[f"{prefix}lr-lasso_cv"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.linear_model.LassoCV()))

    if not exclude_limited_elastic_net:
        experiments[f"{prefix}lr-en-fast_cv_l1-ratio_0.5"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.linear_model.ElasticNetCV(l1_ratio=[0.5])))

    if not exclude_full_elastic_net:
        experiments[f"{prefix}lr-en-fast_cv"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.linear_model.ElasticNetCV(
                    # see ElasticNetCV docs for choice of l1_ratios
                    l1_ratio=np.concatenate([np.linspace(0.1, 1, 10), [0.95, 0.98, 0.99]]))))

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments


def init_default_experiments_regression_optimized_crosssplit_linear(
        preprocessing="standard",
        cross_split_scoring=None,
        cv=None,
        prefix="model_",
        exclude_limited_elastic_net=True,
        exclude_full_elastic_net=True,
        exclude_full_svm=True,
        groups=True,
        return_dict=False):
    """
    Parameters
    ----------
    preprocessing
    prefix
    exclude_limited_elastic_net
    exclude_full_elastic_net
    return_dict

    Returns
    -------

    """

    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # default

    import sklearn.linear_model
    import nalabtools.sklearn.model_selection
    import scipy.stats

    if cross_split_scoring is None:
        cross_split_scoring = dict(score_func=lambda y, y_pred: scipy.stats.spearmanr(y, y_pred)[0], prediction="predict")

    if cv is None:
        if groups:
            cv = sklearn.model_selection.LeaveOneGroupOut()
        else:
            cv = sklearn.model_selection.LeaveOneOut()

    # lasso
    e = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                sklearn.linear_model.Lasso(),
                param_grid=dict(
                    alpha=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                ),
                cross_split_scoring=cross_split_scoring,
                cv=cv
            )
        )
    )
    if groups:
        e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
    experiments[f"{prefix}lr-lasso_cv-crosssplit"] = e

    # ridge
    e = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                sklearn.linear_model.Ridge(),
                param_grid=dict(
                    alpha=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                ),
                cross_split_scoring=cross_split_scoring,
                cv=cv
            )
        )
    )
    if groups:
        e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
    experiments[f"{prefix}lr-ridge_cv-crosssplit"] = e
    
    # linear svm

    e = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                sklearn.svm.SVR(kernel="linear", epsilon=0.1),
                param_grid=dict(
                    # https://stackoverflow.com/questions/26337403/what-is-a-good-range-of-values-for-the-svm-svc-hyperparameters-to-be-explored
                    C=np.logspace(-3, 2, 6),
                ),
                cross_split_scoring=cross_split_scoring,
                cv=cv
            )
        )
    )
    if groups:
        e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
    experiments[f"{prefix}svm-linear_cv-crosssplit_epsilon_0.1"] = e

    if not exclude_full_svm:
        e = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                    sklearn.svm.SVR(kernel="linear"),
                    param_grid=dict(
                        # https://stackoverflow.com/questions/26337403/what-is-a-good-range-of-values-for-the-svm-svc-hyperparameters-to-be-explored
                        C=np.logspace(-3, 2, 6),
                        gamma=np.logspace(-3, 2, 6)
                    ),
                    cross_split_scoring=cross_split_scoring,
                    cv=cv
                )
            )
        )
        if groups:
            e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
        experiments[f"{prefix}svm-linear_cv-crosssplit"] = e

    # elastic net

    if not exclude_limited_elastic_net:
        e = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                    sklearn.linear_model.ElasticNet(),
                    param_grid=dict(
                        alpha=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                        l1_ratio=[0.5],
                    ),
                    cross_split_scoring=cross_split_scoring,
                    cv=cv
                )
            )
        )
        if groups:
            e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
        experiments[f"{prefix}lr-en_cv-crosssplit_l1-ratio_0.5"] = e

    if not exclude_full_elastic_net:
        e = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                nalabtools.sklearn.model_selection.CrossSplitScoreGridSearchCV(
                    sklearn.linear_model.ElasticNet(),
                    param_grid=dict(
                        alpha=np.logspace(-4, 4, 10),  # cf. sklearn `_logistic_regression_path`
                        l1_ratio=np.concatenate([np.linspace(0.1, 1, 10), [0.95, 0.98, 0.99]]), # only for elastic net
                    ),
                    cross_split_scoring=cross_split_scoring,
                    cv=cv
                )
            )
        )
        if groups:
            e["fit_params_groups"] = ["crosssplitscoregridsearchcv__groups"]
        experiments[f"{prefix}lr-en_cv-crosssplit"] = e

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments


def init_default_experiments_classification_random(
        preprocessing="standard",
        opt_n_iter=50,
        opt_kwargs: dict = None,
        include_linear_models=True,
        prefix="model_",
        return_dict=False):
    """
    Notes:
    * We set `tol` and `max_iter` for `LogisticRegression` to `1E-4` and `1E3`, respectively,
      in order to match default regression equivalents.
    * Groups for inner cross validation not supported.

    Parameters
    ----------
    preprocessing
    opt_kwargs
    opt_n_iter
    include_linear_models
    prefix
    return_dict

    Returns
    -------

    """
    experiments = collections.OrderedDict()
    preprocessing = parse_preprocessing(preprocessing)

    # prepare args
    if opt_kwargs is None:
        opt_kwargs = dict()
    if "n_iter" not in opt_kwargs:
        opt_kwargs["n_iter"] = opt_n_iter

    # non-linear models

    """Maria's rant on RF parameters (2020-06-08, Slack, personal message):
    the number of variables tested at each branch for splitting, number of trees and minimum number of samples required 
    for splitting at a node (similar restriction to tree depth)
    RFs are pretty hard to overfit, which is why they're pretty robust to hyperparameter selection. But overall imagine 
    that the higher the number of variables you test for splitting, the more similar the trees are to each other, 
    so you probably need less trees. And depending on how well you want them to fit your training dataset, 
    you would allow them to split further down
    that's what I have at the moment:
    ```
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 400, num = 50)]
    # Number of features to consider at every split
    max_features = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 200)]
    # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 50, num = 20)]
    #max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    ```
    but max_features can be as big as your number of features at most, but it should probably be quite a bit smaller. 
    The smaller it is, the more random the trees
    and because I only have let's say 70-80 samples, I go pretty low, down to 2-5, maybe 2 is too low already
    but when you start with 70 at the root :grimacing:
    if they are similar enough it won't divide anyway
    even if they are more than what you specified as the minimum required for splitting
    """
    kwargs = opt_kwargs.copy()
    kwargs["param_distributions"] = dict(
        n_estimators=[50, 100, 150, 200, 300, 400],
        max_depth=[None, *range(10, 100, 10)],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4, 5],
        max_features=['sqrt', 'log2'],
        bootstrap=[True, False],
    )
    experiments[f"{prefix}rf-100"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.model_selection.RandomizedSearchCV(
                sklearn.ensemble.RandomForestClassifier(), **kwargs)))

    """
    See: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    """
    kwargs = opt_kwargs.copy()
    kwargs["param_distributions"] = dict(
        C=np.logspace(-2, 10, 13),
        gamma=np.logspace(-9, 3, 13)
    )
    experiments[f"{prefix}svm-rbf"] = dict(
        pipeline=sklearn.pipeline.make_pipeline(
            *preprocessing,
            sklearn.model_selection.RandomizedSearchCV(
                sklearn.svm.SVC(), **kwargs)))

    # linear models
    if include_linear_models:

        kwargs = opt_kwargs.copy()
        kwargs["param_distributions"] = dict(C=np.logspace(-4, 4, 20))
        experiments[f"{prefix}lr-ridge"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.model_selection.RandomizedSearchCV(
                    sklearn.linear_model.LogisticRegression(
                        solver="lbfgs",
                        tol=1E-4,
                        max_iter=1E3),
                    **kwargs)))

        kwargs = opt_kwargs.copy()
        kwargs["param_distributions"] = dict(C=np.logspace(-4, 4, 20))
        experiments[f"{prefix}lr-lasso"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.model_selection.RandomizedSearchCV(
                    sklearn.linear_model.LogisticRegression(
                        penalty="l1",
                        solver="saga",
                        tol=1E-4,
                        max_iter=1E3),
                    **kwargs)))

        kwargs = opt_kwargs.copy()
        kwargs["param_distributions"] = dict(C=np.logspace(-4, 4, 20), l1_ratio=np.linspace(0, 1, 10))
        experiments[f"{prefix}lr-en-fast"] = dict(
            pipeline=sklearn.pipeline.make_pipeline(
                *preprocessing,
                sklearn.model_selection.RandomizedSearchCV(
                    sklearn.linear_model.LogisticRegression(
                        penalty="elasticnet",
                        l1_ratio=0.5,
                        solver="saga",
                        tol=1E-4,
                        max_iter=1E3),
                    **kwargs)))

    if not return_dict:
        experiments = [Experiment(name=k, **v) for k, v in experiments.items()]

    return experiments
