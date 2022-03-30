from numpy.core.numeric import cross
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, _check_param_grid
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.metrics import check_scoring
from sklearn.utils.validation import indexable, _check_fit_params
from joblib import Parallel
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone
from collections import defaultdict
from sklearn.utils.fixes import delayed
from itertools import product
from purpleml.sklearn.model_selection.validation.cross_validate import _fit_and_score
from sklearn.model_selection._validation import _insert_error_scores
import numbers
import time
from sklearn.model_selection._validation import _aggregate_score_dicts
from abc import abstractmethod
import numpy as np
import sklearn.preprocessing


class CrossSplitScoreBaseSearchCV(BaseSearchCV):
    """Abstract base class for hyper parameter search with cross-validation.
    """

    @abstractmethod
    def __init__(self, estimator, *, cross_split_scoring=None, n_jobs_search=None, n_jobs_cv=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):

        self.cross_split_scoring = cross_split_scoring
        self.estimator = estimator
        self.n_jobs_search = n_jobs_search
        self.n_jobs_cv = n_jobs_cv
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y)

        # callable
        score = self.scorer_(self.best_estimator_, X, y)
        if self.multimetric_:
            score = score[self.refit]
        return score

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        refit_metric = "score"

        # cross split: need new way of deriving scorer
        # if callable(self.scoring):
        #     scorers = self.scoring
        # elif self.scoring is None or isinstance(self.scoring, str):
        #     scorers = check_scoring(self.estimator, self.scoring)
        # else:
        #     scorers = _check_multimetric_scoring(self.estimator, self.scoring)
        #     self._check_refit_for_multimetric(scorers)
        #     refit_metric = self.refit

        # cross split: simple cross split scorerer derivation
        if isinstance(self.cross_split_scoring, dict):
            cross_split_scorer = make_simple_cross_split_scorer(**self.cross_split_scoring)
        else:
            cross_split_scorer = self.cross_split_scoring

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        # cross split: 
        # cross split scoring will happen in an inner loop, 
        # thus to keep modifications to a minimum, 
        # we simulate one split: 
        # n_splits = cv_orig.get_n_splits(X, y, groups)
        n_splits = 1

        base_estimator = clone(self.estimator)

        parallel = Parallel(
            n_jobs=self.n_jobs_search,
            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            cross_split_scorer=cross_split_scorer,
            fit_params=fit_params,
            n_jobs=self.n_jobs_cv,
            return_train_score=self.return_train_score,
            error_score=self.error_score,
            verbose=self.verbose)

        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(
                    candidate_params, 
                    cv=None,
                    more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                # cross split: fix split
                split = list(cv.split(X, y, groups))

                # cross split: replace fit and score function
                out = parallel(
                    delayed(_fit_and_score_cross_split)(
                        clone(base_estimator),
                        X, y,
                        split=split,
                        parameters=parameters,
                        split_progress=None,
                        candidate_progress=(
                            cand_idx,
                            n_candidates),
                        **fit_and_score_kwargs)
                    for (cand_idx, parameters) 
                    in enumerate(candidate_params))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                
                # cross split: for cross split scoring, we use custom scores,
                # thus, we disable this (should be taken care of by `_fit_and_score_cross_split`)
                # if callable(self.scoring):
                #     _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out,
                    all_more_results)

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            # cross split: I think we will probably never need this, but this should do
            if self.multimetric_:
            # if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        # self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


def _fit_and_score_cross_split(
        estimator,
        X, y,
        split,
        parameters,
        cross_split_scorer,
        fit_params,
        return_train_score,
        error_score,
        n_jobs,
        candidate_progress,
        split_progress,
        verbose):
    
    # TODO: Transforming labels seems to be necessary for scorer? This should be OK, right? 
    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)

    parallel = Parallel(
        n_jobs=n_jobs,
        # pre_dispatch=self.pre_dispatch
    )

    with parallel:
        out = parallel(
            delayed(_fit_and_score)(
                estimator, X, y, 
                cross_split_scorer, 
                train=train, 
                test=test, 
                verbose=verbose,
                parameters=parameters, 
                fit_params=fit_params, 
                return_test_score=False,
                return_train_score=False,
                return_parameters=False, 
                return_n_test_samples=True,
                return_times=True, 
                return_estimator=False,
                return_splits=False,
                return_test_predictions=True,
                return_train_predictions=return_train_score,
                split_progress=(split_idx, len(split)), 
                candidate_progress=None,
                error_score=error_score)
            for (split_idx, (train, test)) in enumerate(split))

        # TODO: parallelize?
        # out = [_fit_and_score(
        #     estimator, X, y, 
        #     cross_split_scorer, 
        #     train=train, 
        #     test=test, 
        #     verbose=verbose,
        #     parameters=parameters, 
        #     fit_params=fit_params, 
        #     return_test_score=False,
        #     return_train_score=False,
        #     return_parameters=False, 
        #     return_n_test_samples=True,
        #     return_times=True, 
        #     return_estimator=False,
        #     return_splits=False,
        #     return_test_predictions=True,
        #     return_train_predictions=return_train_score,
        #     split_progress=(split_idx, len(split)), 
        #     candidate_progress=None,
        #     error_score=error_score
        # ) for (split_idx, (train, test)) in enumerate(split)]
    
    # prepare output
    out = _aggregate_score_dicts(out)
    out["predictions_test"] = _aggregate_score_dicts(out["predictions_test"])
    if return_train_score:
        out["predictions_train"] = _aggregate_score_dicts(out["predictions_train"])

    # compile result
    result = {}
    result["fit_failed"] = np.any(out["fit_failed"])
    result["score_time"] = np.sum(out["score_time"])
    result["fit_time"] = np.sum(out["fit_time"])
    result["n_test_samples"] = np.sum(out["n_test_samples"])

    # calculate scores
    y_true = [y[t] for _, t in split]
    y_pred = out["predictions_test"]

    score_time_start = time.time()
    score = cross_split_scorer(y_true, y_pred)
    result["score_time"] = time.time() - score_time_start

    result["test_scores"] = score
    if return_train_score:
        # TODO: maybe not the best way to do this; would a separate scorer argument be better?
        y_true = [[y[t]] for t, _ in split]
        y_pred = [{k: [v[i]] for k, v in out["predictions_train"].items()} for i in range(len(split))]
        result["split_train_scores"] = [cross_split_scorer(yt, yp) for yt, yp in zip(y_true, y_pred)]
        result["train_scores"] = np.mean(result["split_train_scores"])

    # return output
    return result

def make_simple_cross_split_scorer(
        score_func, *, 
        greater_is_better=True, 
        prediction="predict",
        **kwargs):

    sign = 1 if greater_is_better else -1

    def scorer(y_true, y_pred):
        y_true = np.concatenate(y_true)
        if prediction == "predict":
            y_pred = np.concatenate(y_pred["predict"])
        elif prediction == "predict_proba_1":
            y_pred = np.concatenate(y_pred["predict_proba"])[:,1]
        elif prediction == "predict_proba":
            y_pred = np.concatenate(y_pred["predict_proba"])
        elif prediction == "decision_function_1":
            y_pred = np.concatenate(y_pred["decision_function"])[:,1]
        elif prediction == "decision_function":
            y_pred = np.concatenate(y_pred["decision_function"])
        else:
            pass
        return sign * score_func(y_true, y_pred, **kwargs)

    return scorer


class CrossSplitScoreGridSearchCV(CrossSplitScoreBaseSearchCV):

    _required_parameters = ["estimator", "param_grid"]

    def __init__(self, estimator, param_grid, *, cross_split_scoring=None,
                 n_jobs_search=None, 
                 n_jobs_cv=None,
                 refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(
            estimator=estimator, cross_split_scoring=cross_split_scoring,
            n_jobs_search=n_jobs_search, n_jobs_cv=n_jobs_cv, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))


# class CrossSplitScoreBaseSearchCV(BaseSearchCV):

#     @abstractmethod
#     @_deprecate_positional_args
#     def __init__(
#             self, estimator, *, 
#             cross_split_scoring=None, 
#             n_jobs=None,
#             refit=True, cv=None, verbose=0,
#             pre_dispatch='2*n_jobs', error_score=np.nan,
#             return_train_score=True):

#         self.cross_split_scoring = cross_split_scoring
        
#         self.estimator = estimator
#         self.n_jobs = n_jobs
#         self.refit = refit
#         self.cv = cv
#         self.verbose = verbose
#         self.pre_dispatch = pre_dispatch
#         self.error_score = error_score
#         self.return_train_score = return_train_score

#     @_deprecate_positional_args
#     def fit(self, X, y=None, *, groups=None, **fit_params):
#         """Run fit with all sets of parameters.
#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Training vector, where n_samples is the number of samples and
#             n_features is the number of features.
#         y : array-like of shape (n_samples, n_output) \
#             or (n_samples,), default=None
#             Target relative to X for classification or regression;
#             None for unsupervised learning.
#         groups : array-like of shape (n_samples,), default=None
#             Group labels for the samples used while splitting the dataset into
#             train/test set. Only used in conjunction with a "Group" :term:`cv`
#             instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
#         **fit_params : dict of str -> object
#             Parameters passed to the ``fit`` method of the estimator
#         """
#         estimator = self.estimator
#         # refit_metric = "score"

#         # TODO: derive scorer
#         # if callable(self.scoring):
#         #     scorers = self.scoring
#         # elif self.scoring is None or isinstance(self.scoring, str):
#         #     scorers = check_scoring(self.estimator, self.scoring)
#         # else:
#         #     scorers = _check_multimetric_scoring(self.estimator, self.scoring)
#         #     self._check_refit_for_multimetric(scorers)
#             # refit_metric = self.refit

#         X, y, groups = indexable(X, y, groups)
#         fit_params = _check_fit_params(X, fit_params)

#         cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
#         n_splits = cv_orig.get_n_splits(X, y, groups)

#         base_estimator = clone(self.estimator)

#         parallel = Parallel(
#             n_jobs=self.n_jobs,
#             pre_dispatch=self.pre_dispatch)

#         fit_and_score_kwargs = dict(
#             scorer=None,
#             fit_params=fit_params,
#             return_train_score=False,
#             return_test_score=False,
#             return_n_test_samples=True,
#             return_times=True,
#             return_estimator=False,
#             return_splits=True,
#             return_parameters=False,
#             return_test_predictions=True,
#             return_train_predictions=True,
#             error_score=self.error_score,
#             verbose=self.verbose)

#         results = {}
#         with parallel:
#             all_candidate_params = []
#             all_out = []
#             all_more_results = defaultdict(list)

#             def evaluate_candidates(
#                     candidate_params, 
#                     cv=None,
#                     more_results=None):
#                 cv = cv or cv_orig
#                 candidate_params = list(candidate_params)
#                 n_candidates = len(candidate_params)

#                 if self.verbose > 0:
#                     print("Fitting {0} folds for each of {1} candidates,"
#                           " totalling {2} fits".format(
#                               n_splits, n_candidates, n_candidates * n_splits))

#                 out = parallel(
#                     delayed(_fit_and_score)(
#                         clone(base_estimator),
#                         X, y,
#                         train=train, test=test,
#                         parameters=parameters,
#                         split_progress=(
#                             split_idx,
#                             n_splits),
#                         candidate_progress=(
#                             cand_idx,
#                             n_candidates),
#                         **fit_and_score_kwargs)
#                     for (cand_idx, parameters),
#                         (split_idx, (train, test)) in product(
#                         enumerate(candidate_params),
#                         enumerate(cv.split(X, y, groups))))

#                 if len(out) < 1:
#                     raise ValueError('No fits were performed. '
#                                      'Was the CV iterator empty? '
#                                      'Were there no candidates?')
#                 elif len(out) != n_candidates * n_splits:
#                     raise ValueError('cv.split and cv.get_n_splits returned '
#                                      'inconsistent results. Expected {} '
#                                      'splits, got {}'
#                                      .format(n_splits,
#                                              len(out) // n_candidates))

#                 # # For callable self.scoring, the return type is only know after
#                 # # calling. If the return type is a dictionary, the error scores
#                 # # can now be inserted with the correct key. The type checking
#                 # # of out will be done in `_insert_error_scores`.
#                 # if callable(self.scoring):
#                 #     _insert_error_scores(out, self.error_score)

#                 all_candidate_params.extend(candidate_params)
#                 all_out.extend(out)
#                 if more_results is not None:
#                     for key, value in more_results.items():
#                         all_more_results[key].extend(value)

#                 nonlocal results
#                 # TODO: This needs to be changed and handled
#                 results = self._format_results(
#                     X=X, y=y, groups=groups, fit_params=fit_params,
#                     all_candidate_params, 
#                     n_splits, 
#                     all_out,
#                     all_more_results)
#                 results 

#                 return results

#             self._run_search(evaluate_candidates)

#             # # multimetric is determined here because in the case of a callable
#             # # self.scoring the return type is only known after calling
#             # first_test_score = all_out[0]['test_scores']
#             # self.multimetric_ = isinstance(first_test_score, dict)

#             # # check refit_metric now for a callabe scorer that is multimetric
#             # if callable(self.scoring) and self.multimetric_:
#             #     self._check_refit_for_multimetric(first_test_score)
#             #     refit_metric = self.refit

#         # # For multi-metric evaluation, store the best_index_, best_params_ and
#         # # best_score_ iff refit is one of the scorer names
#         # # In single metric evaluation, refit_metric is "score"
#         # if self.refit or not self.multimetric_:
#         #     # If callable, refit is expected to return the index of the best
#         #     # parameter set.
#         #     if callable(self.refit):
#         #         self.best_index_ = self.refit(results)
#         #         if not isinstance(self.best_index_, numbers.Integral):
#         #             raise TypeError('best_index_ returned is not an integer')
#         #         if (self.best_index_ < 0 or
#         #            self.best_index_ >= len(results["params"])):
#         #             raise IndexError('best_index_ index out of range')
#         #     else:
#         #         self.best_index_ = results["rank_test_%s"
#         #                                    % refit_metric].argmin()
#         #         self.best_score_ = results["mean_test_%s" % refit_metric][
#         #                                    self.best_index_]
#         #     self.best_params_ = results["params"][self.best_index_]

#         if self.refit:
#             # we clone again after setting params in case some
#             # of the params are estimators as well.
#             self.best_estimator_ = \
#                 clone(clone(base_estimator).set_params(**self.best_params_))
#             refit_start_time = time.time()
#             if y is not None:
#                 self.best_estimator_.fit(X, y, **fit_params)
#             else:
#                 self.best_estimator_.fit(X, **fit_params)
#             refit_end_time = time.time()
#             self.refit_time_ = refit_end_time - refit_start_time

#         # Store the only scorer not as a dict for single metric evaluation
#         # self.scorer_ = scorers

#         self.cv_results_ = results
#         self.n_splits_ = n_splits

#         return self


#     def _format_results(
#             self, 
#             X, y, groups, fit_params,
#             candidate_params, 
#             n_splits, 
#             out,
#             more_results=None):
        
        
#         n_candidates = len(candidate_params)
#         results = dict(more_results or {})

#         breakpoint()
#         # out = _aggregate_score_dicts(out)

#         # def _store(key_name, array, weights=None, splits=False, rank=False):
#         #     """A small helper to store the scores/times to the cv_results_"""
#         #     # When iterated first by splits, then by parameters
#         #     # We want `array` to have `n_candidates` rows and `n_splits` cols.
#         #     array = np.array(array, dtype=np.float64).reshape(n_candidates,
#         #                                                       n_splits)
#         #     if splits:
#         #         for split_idx in range(n_splits):
#         #             # Uses closure to alter the results
#         #             results["split%d_%s"
#         #                     % (split_idx, key_name)] = array[:, split_idx]

#         #     array_means = np.average(array, axis=1, weights=weights)
#         #     results['mean_%s' % key_name] = array_means

#         #     if (key_name.startswith(("train_", "test_")) and
#         #             np.any(~np.isfinite(array_means))):
#         #         warnings.warn(
#         #             f"One or more of the {key_name.split('_')[0]} scores "
#         #             f"are non-finite: {array_means}",
#         #             category=UserWarning
#         #         )

#         #     # Weighted std is not directly available in numpy
#         #     array_stds = np.sqrt(np.average((array -
#         #                                      array_means[:, np.newaxis]) ** 2,
#         #                                     axis=1, weights=weights))
#         #     results['std_%s' % key_name] = array_stds

#         #     if rank:
#         #         results["rank_%s" % key_name] = np.asarray(
#         #             rankdata(-array_means, method='min'), dtype=np.int32)

#         # _store('fit_time', out["fit_time"])
#         # _store('score_time', out["score_time"])
#         # # Use one MaskedArray and mask all the places where the param is not
#         # # applicable for that candidate. Use defaultdict as each candidate may
#         # # not contain all the params
#         # param_results = defaultdict(partial(MaskedArray,
#         #                                     np.empty(n_candidates,),
#         #                                     mask=True,
#         #                                     dtype=object))
#         # for cand_idx, params in enumerate(candidate_params):
#         #     for name, value in params.items():
#         #         # An all masked empty array gets created for the key
#         #         # `"param_%s" % name` at the first occurrence of `name`.
#         #         # Setting the value at an index also unmasks that index
#         #         param_results["param_%s" % name][cand_idx] = value

#         # results.update(param_results)
#         # # Store a list of param dicts at the key 'params'
#         # results['params'] = candidate_params

#         # test_scores_dict = _normalize_score_results(out["test_scores"])
#         # if self.return_train_score:
#         #     train_scores_dict = _normalize_score_results(out["train_scores"])

#         # for scorer_name in test_scores_dict:
#         #     # Computed the (weighted) mean and std for test scores alone
#         #     _store('test_%s' % scorer_name, test_scores_dict[scorer_name],
#         #            splits=True, rank=True,
#         #            weights=None)
#         #     if self.return_train_score:
#         #         _store('train_%s' % scorer_name,
#         #                train_scores_dict[scorer_name],
#         #                splits=True)

#         return results


# class CrossSplitScoreGridSearchCV(CrossSplitScoreBaseSearchCV):
 
#     _required_parameters = ["estimator", "param_grid"]

#     @_deprecate_positional_args
#     def __init__(
#             self, 
#             estimator, 
#             param_grid, 
#             *, 
#             cross_split_scoring=None,
#             n_jobs=None, 
#             refit=True, 
#             cv=None,
#             verbose=0, 
#             pre_dispatch='2*n_jobs',
#             error_score=np.nan, 
#             return_train_score=False):
        
#         super().__init__(
#             estimator=estimator, cross_split_scoring=cross_split_scoring,
#             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
#             pre_dispatch=pre_dispatch, error_score=error_score,
#             return_train_score=return_train_score)
#         self.param_grid = param_grid
#         _check_param_grid(param_grid)

#     def _run_search(self, evaluate_candidates):
#         """Search all candidates in param_grid"""
#         evaluate_candidates(ParameterGrid(self.param_grid))
