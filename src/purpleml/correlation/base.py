import warnings

import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats
import scipy as sp
import joblib
from scipy.special import betainc
import collections
import purpleml
from purpleml.utils.misc import handle_nans, rankcolumns


CorrelationResult = collections.namedtuple("Correlation", ["statistics", "pvalues", "overlap"])
SparseCorrelationResult = collections.namedtuple("Correlation", ["statistics", "pvalues", "overlap", "mask"])


def calculate_correlation_pairwise(
        x, y,
        correlation_function="spearman",
        nan_policy="raise",
        homogeneous_data_policy="raise",
        **kwargs):
    """

    Parameters
    ----------
    x
    y
    correlation_function:
        "spearman", "pearson", or custom function
    nan_policy:
        what to do with NaNs, see `nalabtools.utils.misc.handle_nans`
    homogeneous_data_policy:
        what to do when either x or y only contain the same value
        possible values: "ignore", "raise", "warn",
        or provide a default (r, p) tuple that is returned
        or provide a custom function `func(x,y)`
    kwargs
        kwargs for the correlation function

    Returns
    -------

    """

    overlap = np.sum(~np.isnan(x) & ~np.isnan(y))
    x, y = handle_nans(x, y, nan_policy=nan_policy)

    if homogeneous_data_policy is not None and homogeneous_data_policy != "ignore":
        x_nunique = len(np.unique(x[~np.isnan(x)]))
        y_nunique = len(np.unique(y[~np.isnan(y)]))

        if x_nunique <= 1 or y_nunique <= 1:

            if homogeneous_data_policy == "raise":
                if x_nunique <= 1:
                    raise ValueError("X is homogeneous.")
                if y_nunique <= 1:
                    raise ValueError("Y is homogeneous.")
            elif homogeneous_data_policy == "warn":
                if x_nunique <= 1:
                    warnings.warn("X is homogeneous. There will be NaNs in the results.")
                if y_nunique <= 1:
                    warnings.warn("Y is homogeneous. There will be NaNs in the results.")
            elif isinstance(homogeneous_data_policy, tuple):
                return (*homogeneous_data_policy, overlap)
            else:
                # noinspection PyCallingNonCallable
                x, y = homogeneous_data_policy(x, y)

    if isinstance(correlation_function, str):
        if correlation_function == "spearman":
            r, p = scipy.stats.spearmanr(x, y, **kwargs)
            return r, p, overlap
        elif correlation_function == "pearson":
            r, p = scipy.stats.pearsonr(x, y)
            return r, p, overlap
        elif correlation_function == "kendall":
            r, p = scipy.stats.kendalltau(x, y, **kwargs)
            return r, p, overlap
        else:
            raise Exception(f"Unknown correlation function: {correlation_function}")
    else:
        # noinspection PyCallingNonCallable
        return (*correlation_function(x, y, **kwargs), overlap)


def calculate_correlation_loop(
        X,
        rowvars=True,
        correlation_function="spearman",
        nan_policy="raise",
        homogeneous_data_policy="raise",
        symetric=True,
        progress_reporter=None,
        progress_reporter_level="row",
        **kwargs):

    if not rowvars:
        X = X.T

    statistics = np.empty((X.shape[0], X.shape[0]))
    pvalues = np.empty((X.shape[0], X.shape[0]))
    overlaps = np.empty((X.shape[0], X.shape[0]))

    def progress(l, level=None):
        if progress_reporter is None or (level == "column" and progress_reporter_level == "row"):
            for i in l:
                yield i
        elif progress_reporter == "tqdm":
            import tqdm
            for i in tqdm.tqdm(l, desc=level):
                yield i
        elif progress_reporter == "tqdm_auto":
            import tqdm.auto as tqdm
            for i in tqdm.tqdm(l, desc=level):
                yield i
        else:
            for i in progress(l, level):
                yield i

    for i in progress(list(range(X.shape[0])), "row"):

        row1 = X[i, :]

        for j in progress(list(range(X.shape[0])), "column"):

            if not symetric or i >= j:
                row2 = X[j, :]

                r, p, overlap = calculate_correlation_pairwise(
                    row1, row2,
                    correlation_function=correlation_function,
                    nan_policy=nan_policy,
                    homogeneous_data_policy=homogeneous_data_policy,
                    **kwargs)

                statistics[i, j] = r
                pvalues[i, j] = p
                overlaps[i, j] = overlap

    if symetric:
        statistics.T[np.tril_indices(X.shape[0], -1)] = statistics[np.tril_indices(X.shape[0], -1)]
        pvalues.T[np.tril_indices(X.shape[0], -1)] = pvalues[np.tril_indices(X.shape[0], -1)]
        overlaps.T[np.tril_indices(X.shape[0], -1)] = overlaps[np.tril_indices(X.shape[0], -1)]

    return CorrelationResult(statistics, pvalues, overlaps)


def calculate_correlation_parallel(
        data,
        rowvars=True,
        correlation_function="spearman",
        correlation_threshold=None,
        pvalue_threshold=None,
        nan_policy=False,
        mirror=True,
        n_jobs=None,
        homogeneous_data_policy=None,
        verbose=False):

    if not rowvars:
        data = data.T

    n = data.shape[0]
    shape = (n, n)

    result = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_row_corr)(
        row_idx,
        data,
        n,
        correlation_function=correlation_function,
        homogeneous_data_policy=homogeneous_data_policy,
        nan_policy=nan_policy,
        correlation_threshold=correlation_threshold,
        pvalue_threshold=pvalue_threshold,
        verbose=verbose) for row_idx in range(data.shape[0]))

    # init matrices
    correlation_matrix =    sp.sparse.lil_matrix(shape)
    pvalues =               sp.sparse.lil_matrix(shape)
    mask =                  sp.sparse.lil_matrix(shape)
    overlap =               sp.sparse.lil_matrix(shape)

    # fill matrices with calculated values
    if len(result) > 0:
        for row_idx, row in enumerate(result):
            if len(row) > 0:
                correlation_matrix[row_idx, [ col_idx for col_idx,_,_,_ in row ]] = [ correlation  for _,correlation,_,_ in row ]
                pvalues           [row_idx, [ col_idx for col_idx,_,_,_ in row ]] = [ pvalue       for _,_,pvalue,_      in row ]
                overlap           [row_idx, [ col_idx for col_idx,_,_,_ in row ]] = [ overlap      for _,_,_,overlap     in row ]
                mask              [row_idx, [ col_idx for col_idx,_,_,_ in row ]] = 1

    if mirror:
        correlation_matrix[np.tri(n, n, -1).transpose() == 1] = correlation_matrix.transpose()[np.tri(n) == 0]
        pvalues[np.tri(n, n, -1).transpose() == 1] = pvalues.transpose()[np.tri(n) == 0]
        overlap[np.tri(n, n, -1).transpose() == 1] = overlap.transpose()[np.tri(n) == 0]
        mask[np.tri(n, n, -1).transpose() == 1] = mask.transpose()[np.tri(n) == 0]

    return SparseCorrelationResult(
        correlation_matrix.tocsr(),
        pvalues.tocsr(),
        overlap.tocsr(),
        mask.tocsr()
    )


def _row_corr(
        row1_idx, data, n, homogeneous_data_policy,
        correlation_function, correlation_threshold, pvalue_threshold, nan_policy,
        verbose, **kwargs):
    """
    Helper function for `.calculate_correlation_parallel`.
    """

    if verbose and row1_idx % verbose == 0:
        print("{:3.2f}%, {:5d} / {}".format(row1_idx / n * 100, row1_idx, n))

    col1 = data[row1_idx, :]

    def gen():
        for row2_idx in range(data.shape[0]):
            if row2_idx <= row1_idx:

                col2 = data[row2_idx, :]

                print(row1_idx, row2_idx)
                print(col1)
                print(col2)

                correlation, pvalue, overlap = calculate_correlation_pairwise(
                    col1,
                    col2,
                    correlation_function=correlation_function,
                    nan_policy=nan_policy,
                    homogeneous_data_policy=homogeneous_data_policy,
                    **kwargs)

                if (correlation_threshold is None or correlation >= correlation_threshold) \
                        and (pvalue_threshold is None or pvalue <= pvalue_threshold):
                    yield row2_idx, correlation, pvalue, overlap

    return list(gen())


# def correlations(
#         df,
#         correlation_function="pearson",
#         correlation_threshold=None,
#         pvalue_threshold=None,
#         mirror=True,
#         n_jobs=None,
#         handle_homogeneous_data=None,
#         verbose=False):
#
#     if correlation_function == "pearson":
#         correlation_function = scipy.stats.pearsonr
#     elif correlation_function == "spearman":
#         correlation_function = scipy.stats.spearmanr
#
#     # try to covert to dataframe
#     if not isinstance(df, pd.DataFrame):
#         df = pd.DataFrame(df)
#
#     n = df.shape[1]
#     shape = (n, n)
#
#     # init matrices
#     correlation_matrix = np.zeros(shape)
#     pvalue_matrix = np.zeros(shape)
#     mask_matrix = np.zeros(shape)
#     overlap_matrix = np.zeros(shape)
#
#     for col1_idx in range(df.shape[1]):
#
#         col1 = df.iloc[:, col1_idx]
#         col1check = (col1 - col1.values[0]).sum() == 0
#
#         for col2_idx in range(df.shape[1]):
#             if col2_idx <= col1_idx:
#
#                 col2 = df.iloc[:, col2_idx]
#
#                 if handle_homogeneous_data is not None and (col1check or (col2 - col2.values[0]).sum() == 0):
#                     correlation = handle_homogeneous_data["correlation"]
#                     pvalue = handle_homogeneous_data["pvalue"]
#                 else:
#                     correlation, pvalue = correlation_function(col1, col2)
#
#                 overlap = len(col1) - np.isnan(col1 + col2).sum()
#
#                 if (correlation_threshold is None or correlation >= correlation_threshold) \
#                         and (pvalue_threshold is None or pvalue <= pvalue_threshold):
#                     correlation_matrix[col1_idx, col2_idx] = correlation
#                     pvalue_matrix[col1_idx, col2_idx] = pvalue
#                     overlap_matrix[col1_idx, col2_idx] = overlap
#                     mask_matrix[col1_idx, col2_idx] = 1
#
#     if mirror:
#         correlation_matrix[sp.tri(n, n, -1).transpose() == 1] = correlation_matrix.transpose()[sp.tri(n) == 0]
#         pvalue_matrix[sp.tri(n, n, -1).transpose() == 1] = pvalue_matrix.transpose()[sp.tri(n) == 0]
#         overlap_matrix[sp.tri(n, n, -1).transpose() == 1] = overlap_matrix.transpose()[sp.tri(n) == 0]
#         mask_matrix[sp.tri(n, n, -1).transpose() == 1] = mask_matrix.transpose()[sp.tri(n) == 0]
#
#     return {
#         "correlation_matrix": correlation_matrix,
#         "pvalues": pvalue_matrix,
#         "overlap": overlap_matrix,
#         "mask": mask_matrix
#     }


# def p_values(df, correlation_threshold=0.3, correlation="pearson"):
#
#     if correlation == "pearson":
#         correlation = scipy.stats.pearsonr
#
#     correlation_matrix = df.corr()
#     ps = np.ones(correlation_matrix.shape)
#     overlap = np.zeros(correlation_matrix.shape)
#
#     for i1 in range(len(df.columns)):
#         for i2 in range(len(df.columns)):
#             if abs(correlation_matrix.values[i1, i2]) >= correlation_threshold:
#                 df_no_na = df[[df.columns[i1], df.columns[i2]]].dropna()
#                 overlap[i1,i2] = df_no_na.shape[0]
#                 p = correlation(df_no_na.values[:,0], df_no_na.values[:,1])
#                 ps[i1,i2] = p[1]
#
#     return CorrelationResult(
#         correlation_matrix.values,
#         ps,
#         overlap,
#         None
#     )


# def p_values2(df, correlation="pearson", verbose=False):
#
#     if correlation == "pearson":
#         correlation = scipy.stats.pearsonr
#
#     shape = (df.shape[1], df.shape[1])
#     correlation_matrix = np.empty(shape)
#     ps = np.ones(shape)
#     overlap = np.zeros(shape)
#
#     if verbose:
#         n = df.shape[1] ** 2
#         print("Number of correlations to calculate:", n)
#
#     ii = 0
#     for i1 in range(len(df.columns)):
#         for i2 in range(len(df.columns)):
#
#             if verbose:
#                 ii += 1
#                 if ii % verbose == 0:
#                     print("{:.2f}% ({} / {})".format(ii / n * 100, ii, n))
#
#             df_no_na = df[[df.columns[i1], df.columns[i2]]].dropna()
#             overlap[i1,i2] = df_no_na.shape[0]
#             p = correlation(df_no_na.values[:,0], df_no_na.values[:,1])
#             correlation_matrix[i1,i2] = p[0]
#             ps[i1,i2] = p[1]
#
#     return {
#         "correlation_matrix": correlation_matrix,
#         "pvalues": ps,
#         "overlap": overlap
#     }


def matrix_to_list(data, row_names, column_names=None, filter_matrix=None):
    
    if type(data)!=list:
        data = [data]
    if column_names is None:
        column_names = row_names
        
    l = []
    for ri in range(data[0].shape[0]):
        for ci in range(data[0].shape[1]):
            if filter_matrix is not None:
                if filter_matrix[ri, ci]:
                    l.append(((row_names[ri], column_names[ci]), [ d[ri, ci] for d in data ]))
            else:
                l.append(((row_names[ri], column_names[ci]), [ d[ri, ci] for d in data ]))
    return l


def list_to_df(correlation_list):
    return pd.DataFrame([list(pair) + values for pair, values in correlation_list], columns=["c1", "c2", "correlation", "p", "overlap"])




def to_graph(correlation_df, names, p_threshold=None, remove_isolates=False):
    
    # prepare correlation graph with networkx 
    g = nx.Graph()

    # nodes
    g.add_nodes_from(names)
    
    # edges
    g.add_edges_from([ (d1, d2, { 'correlation':corr, 'p':p, 'overlap':overlap }) 
                      for d1, d2, corr, p, overlap in correlation_df.values 
                      if p_threshold is None or p < p_threshold])

    # remove isolates (no correlations)
    if remove_isolates:
        g.remove_nodes_from(list(nx.isolates(g)))
    
    return g


def rank2d(array):
    """Calculates the ranks for the values in each column of an 2d-array."""
    temp = array.argsort(axis=0)
    ranks = np.empty_like(temp)
    for i, col in enumerate(np.transpose(temp)):
        ranks[col, i] = np.arange(len(col))
    return ranks


def rank1d(array):
    """Calculates ranks in a 1d-array"""
    if isinstance(array, list):
        array = np.array(array)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def derive_correlation_function(correlation_function):
    if isinstance(correlation_function, str):
        if correlation_function == "pearson":
            return pearsonr
        elif correlation_function == "spearman":
            return spearmanr
        else:
            raise ValueError(f"Unknown correlation function: '{correlation_function}'")
    else:
        return correlation_function


def calculate_correlation(X, correlation_function="spearman", statistics_only=True, **kwargs):
    """
    Proxy for matrix based correlation functions.
    """
    if isinstance(correlation_function, str):
        if correlation_function == "spearman":
            args = {**dict(pvalues=not statistics_only, overlap=not statistics_only), **kwargs}
            result = purpleml.correlation.base.spearmanr(X, **args)
        elif correlation_function == "pearson":
            args = {**dict(pvalues=not statistics_only, overlap=not statistics_only), **kwargs}
            result = purpleml.correlation.base.pearsonr(X, **args)
        else:
            raise Exception(f"Unknown correlation function: {correlation_function}")
    else:
        # noinspection PyCallingNonCallable
        result = correlation_function(X)

    if statistics_only:
        return result.statistics
    else:
        return result


def spearmanr(matrix, method="columns", pvalues=True, overlap=True):
    """Spearman correlation analogously to numpy.corrcoef except that this calculates uses rowvar=False by default
    and returns a CorrelationResult(statistics, pvalues, overlap)."""

    if np.isnan(matrix).sum() > 0:
        raise Exception("NaN are not supported.")

    if method == "mstats":
        ranks = scipy.stats.mstats.rankdata(matrix, axis=0)
        return purpleml.correlation.base.pearsonr(ranks, pvalues=pvalues, overlap=overlap)
    elif method == "columns":
        ranks = rankcolumns(matrix)
        return purpleml.correlation.base.pearsonr(ranks, pvalues=pvalues, overlap=overlap)
    else:
        raise Exception(f"Unknown method: {method}")


def pearsonr(matrix, pvalues=True, overlap=True):
    """
    Source: https://stackoverflow.com/a/24547964/991496
    """

    # correlations
    r = np.corrcoef(matrix, rowvar=False)

    # pvalue; TODO: did not check
    if pvalues:
        rf = r[np.triu_indices(r.shape[0], 1)]
        df = matrix.shape[0] - 2
        ts = rf * rf * (df / (1 - rf * rf))
        pf = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts))
        p = np.zeros(shape=r.shape)
        p[np.triu_indices(p.shape[0], 1)] = pf
        p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
        # p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    else:
        p = None

    # overlap
    if overlap:
        nan = np.zeros(matrix.shape)
        nan[np.isnan(matrix)] = 0
        nan[~np.isnan(matrix)] = 1
        overlap = np.dot(nan, nan.T)
    else:
        overlap = None

    return CorrelationResult(r, p, overlap)

