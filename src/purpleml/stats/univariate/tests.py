import numpy as np
import pandas as pd

import scipy.stats
import sklearn.preprocessing
import sklearn.metrics


def fisher_exact(d, t):
    m = pd.crosstab(d, t)
    m.index.name = "data"
    m.columns.name = "target"
    fisher = scipy.stats.fisher_exact(m.values)
    return fisher[0], fisher[1], m


def chi2_contingency(d, t, correction=True, lambda_=None):
    m = pd.crosstab(d, t)
    m.index.name = "data"
    m.columns.name = "target"
    chi2 = scipy.stats.chi2_contingency(m.values, correction=correction, lambda_=lambda_)
    return chi2[0], chi2[1], (chi2, m)


def ranksums_bin_num(d, t):
    values = np.sort(np.unique(d))
    t0 = t[d == values[0]]
    t1 = t[d == values[1]]
    ranksums = scipy.stats.ranksums(t0, t1)
    return ranksums[0], ranksums[1], (ranksums, {
        "values": values,
        "mean0": np.mean(t0),
        "mean1": np.mean(t1),
        "median0": np.median(t0),
        "median1": np.median(t1)})


def ranksums_num_bin(d, t):
    return ranksums_bin_num(t, d)


def mannwhitneyu_bin_num(d, t, use_continuity=False, alternative="two-sided"):
    # TODO: I set parameters different from default ... why?
    values = np.sort(np.unique(d))
    t0 = t[d == values[0]]
    t1 = t[d == values[1]]
    res = scipy.stats.mannwhitneyu(t0, t1, use_continuity=use_continuity, alternative=alternative)
    return res[0], res[1], (res, {
        "values": values,
        "mean0": np.mean(t0),
        "mean1": np.mean(t1),
        "median0": np.median(t0),
        "median1": np.median(t1)})


def mannwhitneyu_num_bin(d, t, use_continuity=False, alternative="two-sided"):
    return mannwhitneyu_bin_num(t, d, use_continuity=use_continuity, alternative=alternative)


def max_auc_num_bin(d, t, average="macro", sample_weight=None, max_fpr=None):
    t = sklearn.preprocessing.LabelEncoder().fit_transform(t)
    auc1 = sklearn.metrics.roc_auc_score(
        t, d, average=average, sample_weight=sample_weight, max_fpr=max_fpr)
    auc2 = sklearn.metrics.roc_auc_score(
        np.abs(t - 1), d, average=average, sample_weight=sample_weight, max_fpr=max_fpr)
    ranksums = ranksums_num_bin(d, t)
    return max(auc1, auc2), ranksums[1], {"reverse": auc2 > auc1, "ranksums": ranksums}


def kruskal_cat_num(d, t, nan_policy="propagate"):
    kruskal = scipy.stats.kruskal(*[t[d == v] for v in np.unique(d)], nan_policy=nan_policy)
    return kruskal[0], kruskal[1], kruskal


def pearsonr(d, t):
    res = scipy.stats.pearsonr(d, t)
    return res[0], res[1], res


def spearmanr(d, t, axis=0, nan_policy="propagate"):
    res = scipy.stats.spearmanr(d, t, axis=axis, nan_policy=nan_policy)
    return res[0], res[1], res


def kendalltau(d, t, initial_lexsort=None, nan_policy="propagate", method="auto"):
    res = scipy.stats.kendalltau(d, t, initial_lexsort=initial_lexsort, nan_policy=nan_policy, method=method)
    return res[0], res[1], res
