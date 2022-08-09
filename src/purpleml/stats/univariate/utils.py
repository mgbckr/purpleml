import collections
import numpy as np
import purpleml.utils


def infer_type(data: np.ndarray, default_numeric="num", numeric_binary=True, other_binary=True, nan_policy="raise"):
    data = purpleml.utils.misc.handle_nans(data, nan_policy=nan_policy)
    msk_nan_data = np.isnan(data)
    if np.issubdtype(data.dtype, np.number):
        if numeric_binary and np.unique(data[~msk_nan_data]).size == 2:
            return "bin"
        else:
            return default_numeric
    else:
        if other_binary and np.unique(data[~msk_nan_data]).size == 2:
            return "bin"
        else:
            return "cat"


def calculate_statistics(data, data_type=None, **infer_args):
    # TODO: handle non-numeric
    data_type_inferred = False
    if data_type is None:
        data_type = infer_type(data, **infer_args)
        data_type_inferred = True

    na_idx = np.isnan(data)
    # data = data[~na_idx]
    return collections.OrderedDict(
        n=data.size,  # + np.sum(na_idx),
        n_na=np.sum(na_idx),
        n_unique=np.unique(data).size,
        data_type=data_type,
        data_type_inferred=data_type_inferred,
        mean=np.nanmean(data),
        std=np.nanstd(data),
        median=np.nanmedian(data),
        min=np.nanmin(data),
        max=np.nanmax(data))
