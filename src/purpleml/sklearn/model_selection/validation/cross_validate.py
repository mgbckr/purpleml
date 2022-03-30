# The idea is to extend sklearn's cross_validate function a bit 
# to return everything we could ever need (predictions, models, etc.).
#
# This is a shameless copy from scikit-learn's `cross_validate` in combination with inspiration from: 
# https://github.com/robbisg/scikit-learn/blob/group_cv/sklearn/model_selection/_validation.py
# Based on scikit-learn version 24.2: 
# https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/model_selection/_validation.py 

import numbers
import time
import warnings

import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.model_selection._validation import _check_fit_params
from sklearn.model_selection._validation import _score
from sklearn.utils import indexable
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples
import sklearn.base
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import _normalize_score_results
from sklearn.model_selection._validation import _insert_error_scores
from traceback import format_exc
from joblib import Parallel, logger

__all__ = ['cross_validate', 'cross_validate_repeatedly']

# TODO:
# * add function to ensure same data sets for splits/gridcv? 


class Preprocessor(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def process(self, X, y=None, groups=None, fit_params=None):
        """
        Returns
        -------
        X, y, groups, fit_params
        """
        pass


class PipelinePreprocessor(Preprocessor):

    def __init__(
            self,
            preprocessor_X=None,
            preprocessor_y=None,
            preprocessor_groups=None,
            preprocessor_fit_params=None):
        self.preprocessor_X = preprocessor_X
        self.preprocessor_y = preprocessor_y
        self.preprocessor_groups = preprocessor_groups
        self.preprocessor_fit_params = preprocessor_fit_params

    def process(self, X, y=None, groups=None, fit_params=None):
        if self.preprocessor_X is not None:
            try:
                X = clone(self.preprocessor_X).fit_transform(X=X, y=y, groups=groups)
            except (ValueError, TypeError):
                X = clone(self.preprocessor_X).fit_transform(X=X, y=y)
        if self.preprocessor_y is not None:
            try:
                y = clone(self.preprocessor_y).fit_transform(X=X, y=y, groups=groups)
            except (ValueError, TypeError):
                y = clone(self.preprocessor_y).fit_transform(X=X, y=y)
        if self.preprocessor_groups is not None:
            try:
                groups = clone(self.preprocessor_groups).fit_transform(X=X, y=y, groups=groups)
            except (ValueError, TypeError):
                groups = clone(self.preprocessor_groups).fit_transform(X=X, y=y)
        return X, y, groups, fit_params


def check_numpy(name, array, raise_exception=True):
    if array is not None and not isinstance(array, np.ndarray):
        msg = f"{name} is not a numpy array! (type={type(array)})"
        if raise_exception:
            raise ValueError(msg)
        else:
            warnings.warn(msg)


def cross_validate_repeatedly(
        estimator, X, y=None, groups=None,
        preprocessor=None,
        scoring=None,
        cv='warn',
        n_jobs=None, 
        pre_dispatch='2*n_jobs',
        verbose=0, 
        fit_params=None,
        return_test_score=True,
        return_train_score=False,
        return_estimator=False,
        return_splits=False,
        return_test_predictions=False,
        return_train_predictions=False,
        return_y=False,
        error_score=np.nan,
        allow_non_numpy_data=False,
        # parameters for repetition
        n_repeat=None,
        n_jobs_repeat=None,
        return_style=None, 
        pre_dispatch_repeat='2*n_jobs'):
    
    if n_repeat is None:
        n_repeat = 1

    parallel = Parallel(
        n_jobs=n_jobs_repeat,
        pre_dispatch=pre_dispatch_repeat, 
        verbose=verbose)

    scores = parallel(
        delayed(cross_validate)(
            estimator, X, y=y, groups=groups, 
            scoring=scoring, cv=cv,
            preprocessor=preprocessor,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose, 
            fit_params=fit_params, 
            return_test_score=return_test_score,
            return_train_score=return_train_score,
            return_estimator=return_estimator, 
            return_splits=return_splits,
            return_test_predictions=return_test_predictions,
            return_train_predictions=return_train_predictions,
            return_y=return_y,
            error_score=error_score,
            allow_non_numpy_data=allow_non_numpy_data)
        for _ in range(n_repeat))
    
    if return_style == "merge":
        return {k: [s[k] for s in scores] for k in scores[0].keys()}
        
    elif return_style is None or return_style == "list":
        return scores


def cross_validate(
        estimator, X, y=None, *, groups=None, 
        scoring=None, cv=None,
        preprocessor=None,
        n_jobs=None, verbose=0, 
        fit_params=None,
        pre_dispatch='2*n_jobs', 
        return_test_score=True,
        return_train_score=False,
        return_estimator=False,  
        return_splits=False,
        return_test_predictions=False,
        return_train_predictions=False,
        return_y=False,
        error_score=np.nan,
        allow_non_numpy_data=False):
    """Evaluate metric(s) by cross-validation and also record fit/score times.
    Read more in the :ref:`User Guide <multimetric_cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:
        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.
        If `scoring` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
        See :ref:`multimetric_grid_search` for an example.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
        .. versionadded:: 0.19
        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
        .. versionadded:: 0.20
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
        .. versionadded:: 0.20
    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:
            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    Single metric evaluation using ``cross_validate``
    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.33150734, 0.08022311, 0.03531764])
    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)
    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28010158 0.39088426 0.22784852]
    See Also
    ---------
    cross_val_score : Run cross-validation for single metric evaluation.
    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.
    """

    # pre-processing
    if preprocessor is not None:
        X, y, groups, fit_params = preprocessor.process(X, y, groups, fit_params)

    # check data
    X, y, groups = indexable(X, y, groups)

    # warn
    check_numpy("X", X, raise_exception=not allow_non_numpy_data)
    check_numpy("y", y, raise_exception=not allow_non_numpy_data)
    check_numpy("groups", groups, raise_exception=not allow_non_numpy_data)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator), X, y, scorers, train, test, verbose, 
            None,
            fit_params, 
            return_train_score=return_train_score,
            return_test_score=return_test_score,
            return_times=True, 
            return_estimator=return_estimator,
            return_splits=return_splits,
            return_test_predictions=return_test_predictions,
            return_train_predictions=return_train_predictions,
            error_score=error_score)
        for train, test in cv.split(X, y, groups))

    # For callabe scoring, the return type is only know after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret['fit_time'] = results["fit_time"]
    ret['score_time'] = results["score_time"]

    # estimator
    if return_estimator:
        ret['estimator'] = results["estimator"]

    # splits
    if return_splits:
        if return_splits:
            ret['splits_test'] = results["splits_test"]
            ret['splits_train'] = results["splits_train"]

    # prediction
    if return_test_predictions:
        ret['predictions_test'] = results["predictions_test"]
    if return_train_predictions:
        ret['predictions_train'] = results["predictions_train"]

    # scores

    if return_test_score:
        test_scores_dict = _normalize_score_results(results["test_scores"])
        for name in test_scores_dict:
            key = 'test_%s' % name
            ret[key] = test_scores_dict[name]

    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])
        for name in train_scores_dict:
            key = 'train_%s' % name
            ret[key] = train_scores_dict[name]
    
    # y
    if return_y:
        ret["y"] = y

    return ret


def _fit_and_score(
        estimator, X, y, 
        scorer, train, test, verbose,
        parameters, 
        fit_params, 
        return_test_score=True,
        return_train_score=False,
        return_parameters=False, 
        return_n_test_samples=False,
        return_times=False, 
        return_estimator=False,
        return_splits=False,
        return_test_predictions=False,
        return_train_predictions=False,
        split_progress=None, 
        candidate_progress=None,
        error_score=np.nan):

    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_test_score : bool, default=True
        Compute and return score on testing set.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    return_splits : boolean, default=False
        Whether to return the splits used for calculating scores.
    return_test_predictions : boolean, default=False
        Whether to return test predictions (predict, predict_proba and decision_function).
    return_train_predictions : boolean, default=False
        Whether to return train predictions (predict, predict_proba and decision_function).

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
        splits_train : dict
            The splits used for training the estimator
        splits_test : dict
            The splits used for evaluating the estimator
        predictions_test : dict
            The test predictions (predict, predict_proba and decision_function)
        predictions_train : dict
            The train predictions (predict, predict_proba and decision_function)
    """
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += (f"; {candidate_progress[0]+1}/"
                             f"{candidate_progress[1]}")

    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = (', '.join(f'{k}={parameters[k]}'
                                    for k in sorted_keys))
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                if return_test_score:
                    test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        score_time = 0.0
        if return_test_score:
            test_scores = _score(estimator, X_test, y_test, scorer, error_score)
            score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(
                estimator, X_train, y_train, scorer, error_score
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict) and (return_test_score or return_train_score):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    if return_test_score:
                        scorer_scores = test_scores[scorer_name]
                        result_msg += f"test={scorer_scores:.3f})"
            else:
                msg_scores = []
                if return_train_score:
                    msg_scores.append(f"train={train_scores:.3f}")
                if return_test_score:
                    msg_scores.append(f"test={test_scores:.3f}")
                result_msg += ", score=(" + ", ".join(msg_scores) + ")"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    if return_test_score:
        result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator

    if return_splits:
        result["splits_test"] = test
        result["splits_train"] = train

    if return_test_predictions:
        predictions = {}
        for f in ["predict", "predict_proba", "decision_function"]:
            if hasattr(estimator, f):
                predictions[f] = getattr(estimator, f)(X_test)
        result["predictions_test"] = predictions

    if return_train_predictions:
        predictions = {}
        for f in ["predict", "predict_proba", "decision_function"]:
            if hasattr(estimator, f):
                predictions[f] = getattr(estimator, f)(X_train)
        result["predictions_train"] = predictions
    return result