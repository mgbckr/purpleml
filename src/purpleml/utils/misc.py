import collections
import pathlib
import re
import warnings

import numpy as np
import pandas
import scipy.stats
import matplotlib.pyplot as plt

import sklearn.impute

from functools import wraps
import time

import unicodedata


def group_apply(data, groups, func, axis=0, func_params=None, order=None, return_groups=False):

    if func_params is None:
        func_params = {}

    try:
        groups = [tuple(g) for g in groups]
    except:
        pass
    group_names = list({g for g in groups})

    slice_tuple = lambda g: tuple([[gg == g for gg in groups] if i == axis else slice(None) for i in range(data.ndim)])
    group_results = collections.OrderedDict()
    for g in group_names:
        group_results[g] = func(data[slice_tuple(g)], **func_params)

    if order is None:
        applied = np.array([group_results[g] for g in group_names])
        if return_groups:
            return applied, np.array(group_names)
        else:
            return applied
    elif isinstance(order, str) and order == "dict":
        return group_results
    else:
        try:
            applied = np.array([group_results[e] for e in order])
            if return_groups:
                return applied, np.array(group_names)
            else:
                return applied
        except TypeError:
            raise Exception(f"Don't know what to with order: {order}")


def subplots(n, n_per_break, break_row=True, figsize_single=None, **kwargs):
    n_breaks = int(np.ceil(n / n_per_break))
    n_per_break = min(n_per_break, n)

    if figsize_single is not None:
        if isinstance(figsize_single , int):
            kwargs["figsize"] = (kwargs["figsize"] * n_per_break, kwargs["figsize"] * n_breaks)
        else:
            kwargs["figsize"] = (figsize_single[0] * n_per_break, figsize_single[1] * n_breaks)

    if break_row:
        return plt.subplots(n_breaks, n_per_break, **kwargs)
    else:
        return plt.subplots(n_per_break, n_breaks, **kwargs)


def rankcolumns(a, method="average"):
    b = np.empty_like(a)
    for i in range(a.shape[1]):
        b[:, i] = scipy.stats.rankdata(a[:, i], method=method)
    return b


def select(name, include=None, exclude=None):
    """
    Checks whether name is selected based on an include and an exclude filter.
    If include is defined it is applied first. 
    Exclude is applied only if the name is included.
    """

    # filter pipelines
    if include is not None:
        if isinstance(include, str):
            if re.match(include, name) is None:
                return False
        elif name not in include:
            return False

    if exclude is not None:
        if isinstance(exclude, str):
            if re.match(exclude, name) is not None:
                return False
        elif name in exclude:
            return False

    return True


def convert_to_numpy(data, allow_none=False, converter=None):
    if data is None and allow_none:
        return None
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, pandas.DataFrame) or isinstance(data, pandas.Series):
        return data.values
    elif converter is not None:
        return converter(data)
    else:
        raise ValueError(f"Can not convert type {type(data)} to a numpy array.")


def handle_nans(*args, nan_policy="raise"):
    args = [convert_to_numpy(a) for a in args]
    lengths = [0 if a.ndim == 1 else a.shape[1] for a in args]
    args = [a.reshape(-1, 1) if a.ndim == 1 else a for a in args]

    if len(args) > 1:
        data = np.concatenate(args, axis=1)
    else:
        data = args[0]
    idx_nan = np.isnan(data)
    n_nan = np.sum(idx_nan)

    if nan_policy != "ignore" and n_nan > 0:
        if nan_policy == "raise":
            raise ValueError("Data contains NaNs.")
        elif nan_policy == "warn":
            warnings.warn("Data contains NaNs.")
        elif nan_policy == "mean":
            data = sklearn.impute.SimpleImputer(strategy="mean").fit_transform(data)
        elif nan_policy == "median":
            data = sklearn.impute.SimpleImputer(strategy="median").fit_transform(data)
        elif nan_policy == "omit":
            data = data[np.sum(idx_nan, axis=1) == 0, :]
        else:
            # noinspection PyCallingNonCallable
            data = nan_policy(data)

    results = []
    start_idx = 0
    for l in lengths:
        if l == 0:
            results.append(data[:, start_idx:(start_idx + 1)].flatten())
            start_idx += 1
        else:
            results.append(data[:, start_idx:(start_idx + l)])
            start_idx += l
    if len(results) == 1:
        return results[0]
    else:
        return results


def log_(msg, msg_level=1, msg_indent=0, log_level=-1, log_mode="print", **kwargs):
    if log_mode not in ["print", "tqdm"]:
        raise ValueError(f"Unknown verbose mode: {log_mode}")

    split = msg.splitlines()

    if msg_level <= log_level:
        indent_prefix = ' ' * msg_indent * 2
        if log_mode == 'print':
            for s in split:
                print(indent_prefix + s)
        elif log_mode == 'tqdm':
            raise NotImplementedError()


def init_kwargs(kwargs, **defaults):

    if defaults is None:
        defaults = dict()

    if kwargs is not None:
        return {**defaults, **kwargs}
    else:
        return defaults


def timeit(my_func):
    """Source: https://sanjass.github.io/notes/2019/07/28/timeit_decorator"""
    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        print('"{}" took {:.3f} ms to execute\n'.format(my_func.__name__, (tend - tstart) * 1000))
        return output

    return timed


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    Source: https://stackoverflow.com/a/295466/991496
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
