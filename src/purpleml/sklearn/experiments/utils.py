import collections
import re
import tempfile
import time
import dataclasses
import traceback
import warnings
from typing import Any, Mapping
import pickle

import numpy as np
import pandas as pd
import pathlib
import scipy.stats
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

import mlflowhelper.tracking.collections

from purpleml.utils.misc import select
from purpleml.utils.timer import Timer
from purpleml.sklearn.model_selection import LenientlyStratifiedKFold, StratifiedGroupKFold, ShuffledGroupKFold
from purpleml.sklearn.model_selection.validation.cross_validate import cross_validate_repeatedly


def run_experiment(
        experiment,
        default_experiment=None,
        stats_function="default",
        verbose=0):

    # support new Experiment class
    if isinstance(experiment, Experiment):
        experiment = dataclasses.asdict(experiment)
        experiment = {k: v for k,v in experiment.items() if v is not None}
    if isinstance(default_experiment, Experiment):
        default_experiment = dataclasses.asdict(default_experiment)
        default_experiment = {k: v for k, v in default_experiment.items() if v is not None}

    # handle case when experiment is solely a classifier or a pipeline and not a dict
    if not isinstance(experiment, dict):
        experiment = dict(pipeline=experiment)

    # set statistics function
    if isinstance(stats_function, str):
        if stats_function == "default":
            stats_function = calculate_stats
        else:
            raise Exception(f"Unknown stats function: {stats_function}")

    # init prepared experiment
    if default_experiment is not None:
        prepared_experiment = {**default_experiment, **experiment}
    else:
        prepared_experiment = experiment.copy()

    # handle cv config

    prepared_experiment["cv_config"] = {}

    # set cv_config from default experiment (may still be None afterwards)
    if default_experiment is not None \
            and "cv_config" in default_experiment \
            and default_experiment["cv_config"] is not None:
        prepared_experiment["cv_config"] = default_experiment["cv_config"]

    # incorporate cv_config from experiment
    if "cv_config" in experiment and experiment["cv_config"] is not None:
        # extend cv_config if extend property is set to `True` ...
        if "extend" in experiment["cv_config"] and experiment["cv_config"]["extend"]:
            prepared_experiment["cv_config"].update(experiment["cv_config"])
            del prepared_experiment["cv_config"]["extend"]
        # ... otherwise replace cv_config
        else:
            prepared_experiment["cv_config"] = experiment["cv_config"]

    if "fit_params" in prepared_experiment["cv_config"]:
        raise Exception("Please use `fit_params` in the experiment instead of `fit_params` in `cv_config`.")

    # # preprocess data
    # if prepared_experiment.get("preprocessing", None) is not None:
    #     prepared_experiment["X"] = prepared_experiment["preprocessing"].fit_transform(
    #         prepared_experiment["X"], **prepared_experiment.get("preprocessing_fit_params", {}))
    if prepared_experiment.get("preprocessing", None) is not None:
        raise DeprecationWarning(
            "`preprocessing` in experiment is deprecated. Use `cv_config['preprocessor']` instead.")

    # prepare fit_params
    groups = prepared_experiment.get("groups", None)
    fit_params = prepared_experiment.get("fit_params", None)
    fit_params_groups = prepared_experiment.get("fit_params_groups", None)
    if groups is not None and fit_params_groups is not None:
        if fit_params is None:
            fit_params = dict()
        for param in fit_params_groups:
            fit_params[param] = groups

    # start cross validation
    start_time = time.time()
    results = cross_validate_repeatedly(
        estimator=prepared_experiment["pipeline"],
        X=prepared_experiment["X"],
        y=prepared_experiment["y"],
        groups=groups,
        fit_params=fit_params,
        **prepared_experiment["cv_config"])
    end_time = time.time()
    info = {
        "time_started": start_time,
        "time_finished": end_time
    }

    stats = stats_function(results, verbose=verbose)

    return results, stats, info


def run_experiments(
        experiments,
        default_experiment=None,
        stats_function="default",

        # managing results
        result_container=None,
        include_pipelines=None,
        exclude_pipelines=None,
        overwrite=False,
        expunge=r"^test_.*$",
        pre_experiment_hook=None,
        post_experiment_hook=None,
        save_experiment="set_result_container",
        error_handling="raise",
        error_handling_save_experiment="dont_save",
        shuffle_execution_order=True,
        verbose=0,

        prefix=None):
    """
    Note: dropped the parallel version for simplicity.
    TODO: clean up!

    Parameters
    ----------
    experiments
    default_experiment
    stats_function
    result_container
    include_pipelines
    exclude_pipelines
    overwrite
    expunge
    pre_experiment_hook
    post_experiment_hook
    save_experiment
    error_handling
    error_handling_save_experiment
    shuffle_execution_order
    verbose
    prefix

    Returns
    -------

    """

    # support new Experiment class
    if isinstance(experiments, list):
        experiments = collections.OrderedDict([
            (e.name, {k: v for k, v in dataclasses.asdict(e).items() if v is not None}) for e in experiments
        ])
    if isinstance(default_experiment, Experiment):
        default_experiment = dataclasses.asdict(default_experiment)
        default_experiment = {k: v for k,v in default_experiment.items() if v is not None}

    # set prefix
    if prefix is not None:
        experiments = collections.OrderedDict([
            (f"{prefix}{k}", v) for k, v in experiments.items()
        ])

    # init results container
    if result_container is None:
        result_container = collections.OrderedDict()

    # expunge
    if expunge is not None:
        keys_to_expunge = [k for k in result_container.keys() if re.match(expunge, k)]
        if verbose > 1:
            print("###########################")
            print("expunging:")
        for k in keys_to_expunge:
            if verbose > 1:
                print(f"* {k}")
            del result_container[k]

    # select pipelines
    if verbose > 1:
        print("###########################")
        print("selecting experiments:")

    selected_experiment_names = list()

    existing_exp_names = list(result_container.keys())
    for name, exp in experiments.items():

        select_exp = select(name, include_pipelines, exclude_pipelines)

        redo = True
        if name in existing_exp_names:
            redo = False
            if isinstance(overwrite, bool) and overwrite:
                redo = True
            elif isinstance(overwrite, list) and name in overwrite:
                redo = True
            elif isinstance(overwrite, str) and re.match(overwrite, name) is not None:
                redo = True

            # if not select_exp:
            #     warnings.warn(f"Pipeline skipped: '{name}'. Results are already available.")

        if select_exp and redo:
            selected_experiment_names.append(name)
            if verbose > 1:
                print(f" * [        ] {name}")
        else:
            if verbose > 1:
                if not select_exp:
                    print(f" * [filtered] {name}")
                elif not redo:
                    print(f" * [exists  ] {name}")

    if verbose > 0:
        print("###########################")
        print("running experiments:")

    def run_parallel(name, exp, run=True):

        if not run:
            if verbose > 0:
                print(f"* {name} (SKIPPED)")
            return name, exp, None
        else:
            if verbose > 0:
                print(f"* {name}")

        # init results
        exp_results = {
            "experiment": exp
        }

        try:
            results, stats, info = run_experiment(
                exp,
                default_experiment,
                stats_function=stats_function,
                verbose=verbose)

            exp_results["results"] = results
            exp_results["stats"] = stats
            exp_results["info"] = info

        except Exception as e:
            exp_results = e

        return name, exp, exp_results

    def pre_experiment_processing(exp_name, exp, result_container):

        if pre_experiment_hook == "mlflow-sync":
            if exp_name in result_container and not overwrite:
                return False
            else:
                meta_value = mlflowhelper.tracking.collections.MetaValue(status="RUNNING")
                result_container[exp_name] = meta_value
                return True

        elif pre_experiment_hook is not None:
            return pre_experiment_hook(exp_name, exp, result_container)

        return True

    def post_experiment_processing(exp_name, exp, exp_result, result_container):

        if pre_experiment_hook == "mlflow-sync":
            if isinstance(exp_result, Exception):
                del result_container[exp_name]

        hook_name = ""
        if post_experiment_hook is not None:

            if post_experiment_hook == "persist_immediately":
                result_container[exp_name] = exp_result
                hook_name = "persist_immediately"
            else:
                post_experiment_hook(exp_name, exp, exp_result)
                hook_name = "custom"

        if verbose > 0:
            print(f"* [{hook_name:6s}] {exp_name}")

        # handle exceptions
        if isinstance(exp_result, Exception):
            if error_handling == "raise":
                raise exp_result

            elif error_handling == "warn":
                ex = exp_result
                tb_lines = traceback.format_exception(ex.__class__, ex, ex.__traceback__)
                tb_text = ''.join(tb_lines)
                warnings.warn(f"Error in experiment '{exp_name}': {ex}\n{tb_text}")

            elif error_handling == "ignore":
                pass

            else:
                raise ValueError(f"Unknown error handling strategy: {error_handling}")

    if isinstance(save_experiment, str):
        if save_experiment == "set_result_container":
            if pre_experiment_hook == "mlflow-sync":

                def save_experiment(exp_name, exp, result, result_container):
                    if isinstance(result, Exception):
                        if error_handling_save_experiment == "mlflow-failed":
                            with tempfile.TemporaryDirectory() as dir:
                                error_log = pathlib.Path(dir) / "error_log.txt"
                                with open(pathlib.Path(dir) / "error_log.txt", "w") as file:
                                    tb_lines = traceback.format_exception(
                                        result.__class__, result, result.__traceback__)
                                    tb_text = ''.join(tb_lines)
                                    file.write(tb_text)
                                result_container[exp_name] = \
                                    mlflowhelper.tracking.collections.MetaValue(
                                        result,
                                        status="FAILED",
                                        tags=dict(key="error", value=str(result)),
                                        artifacts=error_log)
                        elif error_handling_save_experiment == "delete":
                            if exp_name in result_container:
                                del result_container[exp_name]
                        elif error_handling_save_experiment == "dont_save":
                            pass
                        else:
                            raise ValueError(
                                f"Unknown error handling strategy for dict entries: '{error_handling_save_experiment}'")
                    elif result is not None:
                        result_container[exp_name] = \
                            mlflowhelper.tracking.collections.MetaValue(result, status="FINISHED")
                    else:
                        warnings.warn("Result was `None`")
                        
            else:
                def save_experiment(exp_name, exp, result, result_container):
                    if isinstance(result, Exception):
                        if error_handling_save_experiment == "delete":
                            if exp_name in result_container:
                                del result_container[exp_name]
                        elif error_handling_save_experiment == "dont_save":
                            pass
                        else:
                            raise ValueError(
                                f"Unknown error handling strategy for dict entries: '{error_handling_save_experiment}'")
                    elif result is not None:
                        result_container[exp_name] = result
                    else:
                        warnings.warn("Result was `None`")
        else:
            raise ValueError(f"Unknown persistence function: '{save_experiment}'")
    elif save_experiment is None:
        def save_experiment(exp_name, exp, result, result_container):
            pass

    if shuffle_execution_order:
        # useful when running `run_experiments` in parallel to avoid conflicts
        np.random.shuffle(selected_experiment_names)

    for name in selected_experiment_names:

        exp = experiments[name]

        run = pre_experiment_processing(name, exp, result_container)
        name, _, exp_result = run_parallel(name, exp, run=run)
        post_experiment_processing(name, exp, exp_result, result_container)
        save_experiment(name, exp, exp_result, result_container)

    return result_container


def calculate_stats(results, verbose=0):
    """Used in `run_experiment`"""
    stats = {}
    for k, v in results[0].items():
        if k.startswith("ensemble_score_"):
            values_all = [r[k] for r in results]
            values = [v for v in values_all if not np.isnan(v)]
            stats[k] = {
                "values": values,
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "n_na": len(values_all) - len(values)}
            if verbose > 0:
                print(f"{k:40s}: {stats[k]['mean']:10.05f} +/- {stats[k]['std']:4f} ({stats[k]['n_na']})")
    return stats


# TODO: This parallel version was intended to be used with the joblib Dask backend in order to better utilize CPUs.
#   However, due to the Dask backend ignoring `n_jobs=1` this causes a heavy over-subscription of CPUs.
#   This was reported as a bug and until this bug is fixed it does not make sense to further develop this parallel
#   version: https://github.com/joblib/joblib/issues/1013
# def run_experiments_parallel(
#         experiments,
#         default_experiment=None,
#         stats_function="default",
#
#         # managing results
#         result_container=None,
#         include_pipelines=None,
#         exclude_pipelines=None,
#         overwrite=False,
#         expunge=r"^test_.*$",
#         pre_experiment_hook=None,
#         post_experiment_hook=None,
#         save_experiment="set_result_container",
#         n_jobs_experiments='sequential',
#         error_handling="raise",
#         verbose=0,
#
#         prefix=None):
#
#     # support new Experiment class
#     if isinstance(experiments, list):
#         experiments = collections.OrderedDict([
#             (e.name, {k: v for k, v in dataclasses.asdict(e).items() if v is not None}) for e in experiments
#         ])
#     if isinstance(default_experiment, Experiment):
#         default_experiment = dataclasses.asdict(default_experiment)
#         default_experiment = {k: v for k,v in default_experiment.items() if v is not None}
#
#     # set prefix
#     if prefix is not None:
#         experiments = collections.OrderedDict([
#             (f"{prefix}{k}", v) for k, v in experiments.items()
#         ])
#
#     # init results container
#     if result_container is None:
#         result_container = collections.OrderedDict()
#
#     # expunge
#     if expunge is not None:
#         keys_to_expunge = [k for k in result_container.keys() if re.match(expunge, k)]
#         if verbose > 1:
#             print("###########################")
#             print("expunging:")
#         for k in keys_to_expunge:
#             if verbose > 1:
#                 print(f"* {k}")
#             del result_container[k]
#
#     # select pipelines
#     if verbose > 1:
#         print("###########################")
#         print("selecting experiments:")
#
#     selected_experiment_names = list()
#
#     for name, exp in experiments.items():
#
#         select_exp = nalabtools.utils.misc.select(name, include_pipelines, exclude_pipelines)
#
#         redo = True
#         if name in result_container:
#             redo = False
#             if isinstance(overwrite, bool) and overwrite:
#                 redo = True
#             elif isinstance(overwrite, list) and name in overwrite:
#                 redo = True
#             elif isinstance(overwrite, str) and re.match(overwrite, name) is not None:
#                 redo = True
#
#             # if not select_exp:
#             #     warnings.warn(f"Pipeline skipped: '{name}'. Results are already available.")
#
#         if select_exp and redo:
#             selected_experiment_names.append(name)
#             if verbose > 1:
#                 print(f" * [        ] {name}")
#         else:
#             if verbose > 1:
#                 if not select_exp:
#                     print(f" * [filtered] {name}")
#                 elif not redo:
#                     print(f" * [exists  ] {name}")
#
#     if verbose > 0:
#         print("###########################")
#         print("running experiments:")
#
#     def run_parallel(name, exp, run=True):
#
#         if not run:
#             if verbose > 0:
#                 print(f"* {name} (SKIPPED)")
#             return name, exp, None
#         else:
#             if verbose > 0:
#                 print(f"* {name}")
#
#         # init results
#         exp_results = {
#             "experiment": exp
#         }
#
#         try:
#             results, stats, info = run_experiment(
#                 exp,
#                 default_experiment,
#                 stats_function=stats_function,
#                 verbose=verbose)
#
#             exp_results["results"] = results
#             exp_results["stats"] = stats
#             exp_results["info"] = info
#
#         except Exception as e:
#             exp_results = e
#
#         return name, exp, exp_results
#
#     def pre_experiment_processing(exp_name, exp, result_container):
#
#         if pre_experiment_hook == "mlflow-sync":
#             if exp_name in result_container and not overwrite:
#                 return False
#             else:
#                 meta_value = mlflowhelper.tracking.collections.MetaValue(status="RUNNING")
#                 result_container[exp_name] = meta_value
#                 return True
#
#         elif pre_experiment_hook is not None:
#             return pre_experiment_hook(exp_name, exp, result_container)
#
#         return True
#
#     def post_experiment_processing(exp_name, exp, exp_result, result_container):
#
#         if pre_experiment_hook == "mlflow-sync":
#             if isinstance(exp_result, Exception):
#                 del result_container[exp_name]
#
#         hook_name = ""
#         if post_experiment_hook is not None:
#
#             if post_experiment_hook == "persist_immediately":
#                 result_container[exp_name] = exp_results
#                 hook_name = "persist_immediately"
#             else:
#                 post_experiment_hook(exp_name, exp, exp_result)
#                 hook_name = "custom"
#
#         if verbose > 0:
#             print(f"* [{hook_name:6s}] {exp_name}")
#
#         # handle exceptions
#         if isinstance(exp_result, Exception):
#             if error_handling == "raise":
#                 raise exp_result
#             elif error_handling == "warn":
#                 warnings.warn(exp_result)
#             elif error_handling == "ignore":
#                 pass
#             else:
#                 raise ValueError(f"Unknown error handling strategy: {error_handling}")
#
#     if isinstance(save_experiment, str):
#         if save_experiment == "set_result_container":
#             if pre_experiment_hook == "mlflow-sync":
#                 def save_experiment(exp_name, exp, result, result_container):
#                     if result is not None:
#                         result_container[exp_name] = \
#                             mlflowhelper.tracking.collections.MetaValue(result, status="FINISHED")
#             else:
#                 def save_experiment(exp_name, exp, result, result_container):
#                     if result is not None:
#                         result_container[exp_name] = result
#         else:
#             raise ValueError(f"Unknown persistence function: '{save_experiment}'")
#     elif save_experiment is None:
#         def save_experiment(exp_name, exp, result, result_container):
#             pass
#
#     if n_jobs_experiments == 'sequential':
#         # Usually joblib should support sequential execution when setting n_jobs=1
#         # but with some backends (dask) it just doesn't care and executes everything at the same time anyway.
#         # So we do it manually.
#
#         for name in selected_experiment_names:
#
#             exp = experiments[name]
#
#             run = pre_experiment_processing(name, exp, result_container)
#             name, _, exp_result = run_parallel(name, exp, run=run)
#             post_experiment_processing(name, exp, exp_result, result_container)
#             save_experiment(name, exp, exp_result, result_container)
#
#     else:
#
#         # For code on immediate results see:
#         # https://stackoverflow.com/questions/38483874/intermediate-results-from-joblib
#         # TODO: this sometimes throws an "OSError: handle is closed" ... not sure how to handle this
#
#         from joblib import register_parallel_backend, parallel_backend
#         from joblib import Parallel, delayed
#
#         active_backend = type(joblib.parallel.get_active_backend()[0])
#
#         class MultiCallback:
#             def __init__(self, *callbacks):
#                 self.callbacks = [cb for cb in callbacks if cb]
#
#             def __call__(self, out):
#                 for cb in self.callbacks:
#                     cb(out)
#
#         class ImmediateResultBackend(active_backend):
#             def callback(self, result):
#                 for name, exp, exp_results in result.result():
#                     # TODO: post processing is only guaranteed for the first failing experiment;
#                     #  neither consequent tasks in the same batch
#                     #  NOR tasks from other batches will get into their postprocessing routine;
#                     #  there is probably no nice way around this
#                     post_experiment_processing(name, exp, exp_result, result_container)
#
#             def apply_async(self, func, callback=None):
#                 cbs = MultiCallback(callback, self.callback)
#                 name, exp = func[1]
#                 run = pre_experiment_processing(name, exp, result_container)
#                 func[1][2] = run
#                 return super().apply_async(func, cbs)
#
#         # register new backend
#         if verbose > 1:
#             print(f"Using backend: {active_backend}")
#         register_parallel_backend('__immediate__', ImmediateResultBackend)
#
#         with parallel_backend('__immediate__'):
#
#             results = Parallel(n_jobs=n_jobs_experiments)(
#                 delayed(run_parallel)(name, experiments[name], True)
#                 for name in selected_experiment_names)
#
#             for name, exp, exp_results in results:
#                 save_experiment(name, exp, exp_results, result_container)
#
#         # clean up backends
#         del joblib.parallel.BACKENDS["__immediate__"]
#
#     return result_container


@dataclasses.dataclass
class Experiment:

    name: str

    X: np.ndarray = None
    X_names: Any = None

    y: np.ndarray = None
    y_name: Any = None

    groups: np.ndarray = None

    pipeline: Any = None
    fit_params: Any = None
    fit_params_groups: Any = None

    cv_config: Any = None

    other: dict = dataclasses.field(default_factory=dict)


def merge_experiments(*experiments):
    merged = experiments[0]
    for e in experiments[1:]:
        if e is not None:
            merged = dataclasses.replace(merged, **{k: v for k, v in dataclasses.asdict(e).items() if v is not None})
    return merged


def cross_experiments(experiments1, experiments2, default_experiment=None):
    derived_experiments = []
    for exp1 in experiments1:
        for exp2 in experiments2:
            if default_experiment is not None:
                merged_exp = merge_experiments(default_experiment, exp1)
            else:
                merged_exp = exp1
            merged_exp = merge_experiments(merged_exp, exp2)
            derived_experiments.append(dataclasses.replace(merged_exp, name=f"{exp1.name}___{exp2.name}"))
    return derived_experiments


def init_experiments(experiments1, experiments2, **kwargs):
    derived_experiments = []
    for exp1 in experiments1:
        exp_list = experiments2(exp1, **kwargs)
        for exp2 in exp_list:
            merged = merge_experiments(exp1, exp2)
            derived_experiments.append(dataclasses.replace(merged, name=f"{exp1.name}___{exp2.name}"))
    return derived_experiments


def extract_predictions(
        results,
        filter_include=None,
        filter_exclude=None,
        verbose=1,
        verbose_progress=False,
        skip_empty_experiments=False,
        cache_file=None,
        max_n_load=None,
        test_predictions=True):
    """

    Parameters
    ----------
    results
    filter_include
    filter_exclude
    verbose
    skip_empty_experiments

    Returns
    -------
    pandas.DataFrame with prediction
    """

    # TODO: enable single experiment and non-repeated

    if cache_file is not None:

        cache_file = pathlib.Path(cache_file)
        if cache_file.exists():
            print(f"Cache file exists; loading: {cache_file}")
            cache = pickle.load(open(cache_file, "rb"))
            cached_experiment_names = cache["experiment"].unique().astype(str)
            if verbose >= 1:
                print(f"Cached experiments: {len(cached_experiment_names)}")
        else:
            warnings.warn(f"Cache file does not exist; will be created: {cache_file}")
            cache = None
            cached_experiment_names = []
    else:
        cache = None
        cached_experiment_names = []

    merged = collections.OrderedDict()

    def add(key, values, batch):
        if key not in merged:
            merged[key] = []
        merged[key].append((batch, values))

    if verbose_progress:
        import tqdm
        result_keys = tqdm.tqdm(list(results.keys()))
    else:
        result_keys = list(results.keys())

    batch = 0
    batch_sizes = []
    experiment_name_id_map = collections.OrderedDict()
    next_experiment_name_id = 0
    loaded = 0
    # print(result_keys)
    for i, exp_name in enumerate(result_keys):

        if select(exp_name, filter_include, filter_exclude) \
                and exp_name not in cached_experiment_names:

            # log
            if verbose_progress:
                result_keys.set_description(exp_name)
            if verbose >= 2:
                print(f"* {i}/{len(results)} {exp_name}")

            # get results
            exp_results = results[exp_name]
            if exp_results is None:
                if skip_empty_experiments:
                    warnings.warn(f"Experiment is empty; skipping: {exp_name}")
                else:
                    raise ValueError(f"Experiment is empty: {exp_name}")
            else:

                # track loaded experiments
                loaded += 1
                if max_n_load is not None and loaded > max_n_load:
                    if verbose >= 1:
                        print(f"Loaded batch of experiments;  stopping: {max_n_load}")
                    break

                for i_repetition, repetition in enumerate(exp_results["results"]):

                    if test_predictions:
                        splits = zip(repetition["splits_test"], repetition["predictions_test"])
                    else:
                        splits = zip(repetition["splits_train"], repetition["predictions_train"])

                    for i_split, (split, predictions) in enumerate(splits):

                        exp_id = experiment_name_id_map.setdefault(exp_name, next_experiment_name_id)
                        if exp_id == next_experiment_name_id:
                            next_experiment_name_id += 1

                        add("experiment", np.repeat(exp_id, split.size), batch=batch)
                        add("repetition", np.repeat(i_repetition, split.size), batch=batch)
                        add("split", np.repeat(i_split, split.size), batch=batch)
                        add("sample_idx", split, batch=batch)
                        add("y", repetition["y"][split], batch=batch)

                        for prediction_name, prediction_data in predictions.items():
                            if len(prediction_data.shape) == 1:
                                add(f"{prediction_name}", prediction_data, batch=batch)
                            else:
                                for i_dim in range(prediction_data.shape[1]):
                                    add(f"{prediction_name}_{i_dim}", prediction_data[:, i_dim], batch=batch)

                        batch += 1
                        batch_sizes.append(split.size)

    for key in list(merged.keys()):
        key_data = []

        batch_start = 0
        batch_end = 0
        i_active = 0

        for i_batch, batch_size in enumerate(batch_sizes):

            batch_end += batch_sizes[i_batch]

            if i_active < len(merged[key]) and merged[key][i_active][0] == i_batch:
                batch, batch_data = merged[key][i_active]
                batch_size = batch_sizes[batch]
                if batch_start + batch_size != batch_end:
                    fill = np.empty(batch_end - batch_start - batch_size)
                    fill[:] = np.nan
                    key_data.append(fill)

                key_data.append(batch_data)
                batch_start = batch_end
                i_active += 1

        end = sum(batch_sizes)
        if batch_start != end:
            fill = np.empty(end - batch_start)
            fill[:] = np.nan
            key_data.append(fill)

        merged[key] = np.concatenate(key_data)

    # if we added any new experiments add them to the cache
    if loaded > 0:  
        
        df = pd.DataFrame(merged)
        df["experiment"] = pd.Categorical.from_codes(df["experiment"], categories=experiment_name_id_map.keys())

        if cache_file is not None:
            if verbose >= 1:
                print(f"New experiments: {df['experiment'].unique().size}")
                print(f"Saving cache: {cache_file}")
            df = pd.concat([cache, df])
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(df, open(cache_file, "wb"))

    else:
        df = cache

    return df


def derive_scorer(scoring):

    if isinstance(scoring, str):

        negative = False
        if re.match(".*_neg$", scoring):
            scoring = scoring.replace("_neg", "")
            negative = True

        if scoring == "spearman r":
            def scorer(y_true, y_pred):
                return scipy.stats.spearmanr(y_pred, y_true)[0]

        elif scoring == "spearman p-value":
            def scorer(y_true, y_pred):
                return scipy.stats.spearmanr(y_pred, y_true)[1]

        elif scoring == "corrected spearman p-value":
            def scorer(y_true, y_pred):
                s, p = scipy.stats.spearmanr(y_pred, y_true)
                return 1 if s < 0 else p

        elif scoring == "ranksums p-value":
            def scorer(y_true, y_pred):
                labels = sorted(np.unique(y_true))
                if len(labels) > 2:
                    raise ValueError("More than two classes given.")
                s, p = scipy.stats.ranksums(y_pred[y_true == labels[0]], y_pred[y_true == labels[1]])
                return p

        elif scoring == "corrected ranksums p-value":

            def scorer(y_true, y_pred, correct="roc_auc"):
                labels = sorted(np.unique(y_true))
                if len(labels) > 2:
                    raise ValueError("More than two classes given.")
                s, p = scipy.stats.ranksums(y_pred[y_true == labels[0]], y_pred[y_true == labels[1]])

                if correct == "roc_auc":
                    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
                    if roc_auc < 0.5:
                        return 1
                    else:
                        return p
                else:
                    if not correct(y_true, y_pred):
                        return 1
                    else:
                        return p

        elif scoring in dir(sklearn.metrics):
            scorer = getattr(sklearn.metrics, scoring)

        elif f"{scoring}_score" in dir(sklearn.metrics):
            scorer = getattr(sklearn.metrics, f"{scoring}_score")

        else:
            raise ValueError(f"Unknown score function: '{scoring}'")

        if negative:

            scorer_nonnegative = scorer

            def scorer(y_true, y_pred, **kwargs):
                return -scorer_nonnegative(y_true, y_pred, **kwargs)

    else:
        scorer = scoring

    return scorer


def calculate_score(
        predictions,
        scoring="spearman r",
        scoring_kwargs=None,
        scoring_mode="across splits",
        handle_multiple_predictions="raise",
        value_order="predict",
        filter_include=None,
        filter_exclude=None):

    with Timer(name="Top"):
        # derive scoring
        if isinstance(scoring, Mapping):
            print(f"Multiple scores detected: {scoring.keys()}")
            scorer = collections.OrderedDict([(k,derive_scorer(v)) for k,v in scoring.items()])
            if scoring_kwargs is None:
                scoring_kwargs = dict()
        else:
            scorer = dict(score=derive_scorer(scoring))
            if scoring_kwargs is None:
                scoring_kwargs = dict()

        # copy prediction
        experiments = [e for e in predictions["experiment"].unique()
                    if select(e, include=filter_include, exclude=filter_exclude)]
        pred = predictions[predictions["experiment"].isin(experiments)].copy()

        # handle duplicate sample indices

        if scoring_mode == "across splits":
            level = ["experiment", "repetition"]
        elif scoring_mode == "within splits":
            level = ["experiment", "repetition", "split"]
        else:
            raise ValueError(f"Unknown scoring mode: {scoring_mode}")

        groupby = level + ["sample_idx"]
        if pred.groupby(groupby).size().max() > 1:
            if handle_multiple_predictions == "raise":
                raise ValueError("Multiple predictions per sample!")
            elif handle_multiple_predictions == "mean":
                pred = pred.groupby(groupby).mean()
            elif handle_multiple_predictions == "median":
                pred = pred.groupby(groupby).median()
            else:
                raise ValueError(f"Unknown mode for handling duplicate predictions: {handle_multiple_predictions}")
        else:
            pred.set_index(groupby, inplace=True)
        pred.sort_index(inplace=True)

        # set values to go by
        # TODO: need to be more careful with actual nans (returned by the predictor), I guess;
        #       otherwise we might be mixing stuff;
        #       possible solution: do this in the grouping step

        if value_order == "classification":
            value_order = ["predict_proba_1", "decision_function", "predict"]
        elif value_order == "regression" or value_order == "predict":
            value_order = ["predict"]

        pred["value"] = np.nan
        for v in value_order:
            if v in pred.columns:
                pred.loc[pred["value"].isna(), "value"] = pred.loc[pred["value"].isna(), v]

    # calculate scores
    with Timer(name="Last group-by"):
        data = pred. \
            groupby(level=level). \
            apply(lambda x: pd.DataFrame(collections.OrderedDict([
                (k, [v(x["y"], x["value"], **scoring_kwargs.get(k, {}))])
                for k, v in scorer.items()
            ]))).droplevel(-1, axis=0)

    return data


def get_default_config(groups=True, n_splits=10):

    if groups:
        cv = ShuffledGroupKFold(n_splits=n_splits, shuffle=True)
    else:
        cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)

    return {
        "cv": cv,
        "return_test_score": False,
        # "return_ensemble_score": False,
        # "return_ensemble_predictions": False,
        # "return_ensemble_split": False,
        # "return_ensemble_score": True,
        # "return_ensemble_predictions": True,
        # "return_ensemble_split": True,
        "n_repeat": 10,
        "n_jobs_repeat": 5,
        "return_estimator": False,
        "return_train_predictions": True,
        "return_test_predictions": True,
        "return_splits": True,
        "return_y": True
    }


def get_default_config_regression(groups=True, n_splits=10, scoring=False):

    if groups:
        cv = ShuffledGroupKFold(n_splits=n_splits, shuffle=True)
    else:
        cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)

    config = {
        "cv": cv,
        "scoring": {
            "r2": "r2",
            "mean_absolute_error_neg": "neg_mean_absolute_error",
            "mean_squared_error_neg": 'neg_mean_squared_error',
            "spearman_r": sklearn.metrics.make_scorer(
                lambda x, y: scipy.stats.spearmanr(x, y)[0], greater_is_better=True),
            "spearman_p_neg": sklearn.metrics.make_scorer(
                lambda x, y: scipy.stats.spearmanr(x, y)[1], greater_is_better=False),
            "kendalltau": sklearn.metrics.make_scorer(
                lambda x, y: scipy.stats.kendalltau(x, y)[0], greater_is_better=True),
            "kendalltau_p_neg": sklearn.metrics.make_scorer(
                lambda x, y: scipy.stats.kendalltau(x, y)[1], greater_is_better=False),
            "pearson_r": sklearn.metrics.make_scorer(
                lambda x, y: scipy.stats.pearsonr(x, y)[0], greater_is_better=True),
            "pearson_p_neg": sklearn.metrics.make_scorer(
                lambda x, y: scipy.stats.pearsonr(x, y)[1], greater_is_better=False),
        }
    }
    
    config = {**get_default_config(n_splits=n_splits), **config}
    if not scoring:
        del config["scoring"]
    return config


def get_default_config_classification(groups=True, n_splits=10, lenient_stratification=False, scoring=False):

    if groups:
        if lenient_stratification:
            cv = LenientlyStratifiedGroupKFold(n_splits=n_splits, shuffle=True)
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    else:
        if lenient_stratification:
            cv = LenientlyStratifiedKFold(n_splits=n_splits, shuffle=True)
        else:
            cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)

    config = {
        "cv": cv,
        "scoring": {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            # the following do not work with class labels other than 0/1 (see
            # `pos_label` in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
            # "f1": "f1",
            # "precision": "precision",
            # "recall": "recall"
        }
    }

    config = {**get_default_config(n_splits=n_splits), **config}
    if not scoring:
        del config["scoring"]
    return config


def get_default_model_regression_en_cv(groups=True, cv=10):

    if groups:
        if cv == "loo":
            cv = sklearn.model_selection.LeaveOneGroupOut()
        elif isinstance(cv, int):
            cv = ShuffledGroupKFold(n_splits=cv, shuffle=True)

    else:
        if cv == "loo":
            cv = sklearn.model_selection.LeaveOneOut()
        elif isinstance(cv, int):
            cv = sklearn.model_selection.KFold(n_splits=cv, shuffle=True)

    return sklearn.model_selection.GridSearchCV(
        sklearn.linear_model.ElasticNet(tol=1E-5, max_iter=10000),
        iid=True,
        cv=cv,
        param_grid={
            "alpha": np.arange(0.1, 1.1, 0.1),
            "l1_ratio": np.arange(0.1, 1.1, 0.1),
        })


def get_default_model_classification_en_cv(groups=True, cv=10):

    if groups:
        if cv == "loo":
            cv = sklearn.model_selection.LeaveOneGroupOut()
        elif isinstance(cv, int):
            cv = StratifiedGroupKFold(n_splits=cv, shuffle=True)
    else:
        if cv == "loo":
            cv = sklearn.model_selection.LeaveOneOut()
        elif isinstance(cv, int):
            cv = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=True)

    return sklearn.model_selection.GridSearchCV(
        sklearn.linear_model.SGDClassifier(tol=1E-5, max_iter=10000, penalty="elasticnet", loss="log"),
        iid=True,
        cv=cv,
        param_grid={
            "alpha": np.arange(0.1, 1.1, 0.1),
            "l1_ratio": np.arange(0.1, 1.1, 0.1),
        })


def _parse_pipeline(estimator, preprocessing="scale", param_grid=None, gridsearch_params=None):

    if preprocessing is None:
        preprocessing = []
    elif isinstance(preprocessing, str):
        preprocessing = [preprocessing]

    # parse preprocessing
    components = []
    for p in preprocessing:
        if isinstance(p, str):
            if p == "scale":
                p = sklearn.preprocessing.StandardScaler()
            elif p.startswith("pca"):
                pca = p.split("-")
                pca_n = int(pca[1]) if len(pca) > 1 else None
                p = sklearn.decomposition.PCA(n_components=pca_n)
            elif p.startswith("power"):
                p = sklearn.preprocessing.PowerTransformer()
            elif p.startswith("poly"):
                p = sklearn.preprocessing.PolynomialFeatures()
            elif p.startswith("inter2"):
                p = sklearn.preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
            else:
                raise Exception(f"Unknown preprocessing method: '{p}'")
        components.append(p)

    # add estimator
    components.append(estimator)

    # create pipeline
    pipeline = sklearn.pipeline.make_pipeline(*components)

    # grid search
    if param_grid is not None:
        if gridsearch_params is None:
            gridsearch_params = {}
        pipeline = sklearn.model_selection.GridSearchCV(pipeline, param_grid, **gridsearch_params)

    return pipeline


def get_default_pipeline_classification(
        classifier,
        classifier_params=None,
        preprocessing="scale",
        groups=True,
        param_grid=None,
        gridsearch_params=None):

    if classifier_params is None:
        classifier_params = {}

    if isinstance(classifier, str):
        if classifier.startswith("rf"):
            rf = classifier.split("-")
            rf_n = int(rf[1]) if len(rf) > 1 else None
            classifier = sklearn.ensemble.RandomForestClassifier(**{**dict(n_estimators=rf_n), **classifier_params})
        elif classifier == "en_cv":
            classifier = get_default_model_classification_en_cv(groups=groups, **classifier_params)
        elif classifier == "feature":
            classifier = nalabtools.sklearn.estimator.FeatureClassifier(**classifier_params)
        elif classifier == "logistic":
            classifier = sklearn.linear_model.LogisticRegression(**classifier_params)
        else:
            raise Exception(f"Unknown classifier: {classifier}")
    else:
        classifier = classifier(**classifier_params)

    return _parse_pipeline(
        classifier, preprocessing=preprocessing, param_grid=param_grid, gridsearch_params=gridsearch_params)


def get_default_pipeline_regression(regressor, regressor_params=None, preprocessing="scale", groups=True):

    if regressor_params is None:
        regressor_params = {}

    if isinstance(regressor, str):
        if regressor.startswith("rf"):
            rf = regressor.split("-")
            rf_n = int(rf[1]) if len(rf) > 1 else 100
            regressor = sklearn.ensemble.RandomForestRegressor(**{**dict(n_estimators=rf_n), **regressor_params})
        elif regressor == "en_cv":
            regressor = get_default_model_regression_en_cv(groups=groups, **regressor_params)
    return _parse_pipeline(regressor, preprocessing=preprocessing)


class FilterIndexTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, idx=None, which="X"):
        self.idx = idx
        self.which = which

    def fit(self, X, y=None, groups=None, idx=None, **kwargs):
        if idx is not None:
            self.idx = idx
        return self

    def transform(self, X, y=None, groups=None, idx=None, invert=False):

        if idx is None:
            idx = self.idx
        if idx is not None:
            idx = check_array(idx, ensure_2d=False, dtype=None).reshape(-1)

        X = check_array(X)
        y = None if y is None else check_array(y, ensure_2d=False, dtype=None)
        groups = None if groups is None else check_array(groups, ensure_2d=False, dtype=None)

        if self.which == "X":
            if idx is None:
                return X
            else:
                return X[idx, :]
        elif self.which == "y":
            if idx is None:
                return y
            else:
                return y[idx]
        elif self.which == "groups":
            if idx is None:
                return groups
            else:
                return groups[idx]
        else:
            raise Exception(f"Unknown 'which': {self.which}")

    def fit_transform(self, X, y=None, groups=None, idx=None):
        return self.fit(X=X, y=y, groups=groups, idx=idx).transform(X=X, y=y, groups=groups, idx=idx)
