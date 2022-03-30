import collections
from collections import Mapping
import warnings

import pandas as pd
import numpy as np

import scipy.stats
import sklearn.preprocessing
import statsmodels.stats.multitest

from purpleml.utils.misc import handle_nans, convert_to_numpy, select
from purpleml.utils.dicts import recursive_merge


def infer_type(data: np.ndarray, default_numeric="num", numeric_binary=True, other_binary=True, nan_policy="raise"):
    data = handle_nans(data, nan_policy=nan_policy)
    idx_nan_data = np.isnan(data)
    if np.issubdtype(data.dtype, np.number):
        if numeric_binary and np.unique(data[~idx_nan_data]).size == 2:
            return "bin"
        else:
            return default_numeric
    else:
        if other_binary and np.unique(data[~idx_nan_data]).size == 2:
            return "bin"
        else:
            return "cat"


def calculate_statistics(data, data_type=None, **infer_args):
    # TODO: handle non-numeric
    if data_type is None:
        data_type = infer_type(data, **infer_args)

    na_idx = np.isnan(data)
    # data = data[~na_idx]
    return collections.OrderedDict(
        n=data.size,  # + np.sum(na_idx),
        n_na=np.sum(na_idx),
        n_unique=np.unique(data).size,
        data_type=data_type,
        mean=np.nanmean(data),
        std=np.nanstd(data),
        median=np.nanmedian(data),
        min=np.nanmin(data),
        max=np.nanmax(data))


def calculate_pairwise_comparison(
        data,
        target,
        data_type=None,
        target_type=None,
        tests=None,
        test_kws=None,
        test_include=None,
        test_exclude=None,
        nan_policy_infer="raise",
        nan_policy_stats="raise",
        handle_error="raise",
        infer_type_before_handling_nans=True):

    def flip(d, t, f, **kwargs):
        return f(t, d, **kwargs)

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
        return flip(d, t, ranksums_bin_num)

    # TODO: I set parameters different from normal ... why?
    def mannwhitneyu_bin_num(d, t, use_continuity=False, alternative="two-sided"):
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
        return flip(d, t, mannwhitneyu_bin_num, use_continuity=use_continuity, alternative=alternative)

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

    predefined_tests = collections.OrderedDict([
        (("bin", "bin"), [
            ("fisher_exact", fisher_exact),
            ("chi2", chi2_contingency)
        ]),
        (("bin", "num"), [
            ("ranksums", ranksums_bin_num),
            ("mannwhitneyu", mannwhitneyu_bin_num),
        ]),
        (("bin", "ord"), [
            ("ranksums", ranksums_bin_num),
            ("mannwhitneyu", mannwhitneyu_bin_num),
        ]),
        (("cat", "bin"), [
            ("chi2", chi2_contingency)
        ]),
        (("ord", "bin"), [
            ("ranksums", ranksums_num_bin),
            ("mannwhitneyu", mannwhitneyu_num_bin),
        ]),
        (("ord", "num"), [
            ("pearson", pearsonr),
            ("spearman", spearmanr),
            ("kendall", kendalltau)
        ]),
        (("num", "bin"), [
            ("ranksums", ranksums_num_bin),
            ("mannwhitneyu", mannwhitneyu_num_bin),
            ("max_auc", max_auc_num_bin),
        ]),
        (("cat", "num"), [
            ("kruskal", kruskal_cat_num),
        ]),
        (("num", "ord"), [
            ("pearson", pearsonr),
            ("spearman", spearmanr),
            ("kendall", kendalltau)
        ]),
        (("num", "num"), [
            ("pearson", pearsonr),
            ("spearman", spearmanr),
            ("kendall", kendalltau)
        ]),
    ])
    tests = predefined_tests if tests is None \
        else recursive_merge(predefined_tests, tests, merge_lists=True)

    if test_kws is None:
        test_kws = {}

    # convert to numpy
    data = convert_to_numpy(data)
    target = convert_to_numpy(target)

    # collect basic stats
    overlap_n = np.sum(~np.isnan(data) & ~np.isnan(target))

    if infer_type_before_handling_nans:
        if data_type is None:
            data_type = infer_type(data, nan_policy=nan_policy_infer)
        if target_type is None:
            target_type = infer_type(target, nan_policy=nan_policy_infer)

    # handle nans
    data, target = handle_nans(data, target, nan_policy=nan_policy_stats)

    if not infer_type_before_handling_nans:
        if data_type is None:
            data_type = infer_type(data, nan_policy=nan_policy_infer)
        if target_type is None:
            target_type = infer_type(target, nan_policy=nan_policy_infer)

    # run tests
    result = collections.OrderedDict()
    test_type = (data_type, target_type)
    if test_type in tests:
        tests = tests[test_type]
        for test_name, test in tests:
            if select(test_name, include=test_include, exclude=test_exclude):
                # noinspection PyBroadException
                result[test_name] = collections.OrderedDict(overlap=overlap_n)
                try:
                    s, p, d = test(data, target)
                    result[test_name]["statistic"] = s
                    result[test_name]["pvalue"] = p
                    result[test_name]["data"] = d
                except Exception as e:
                    if handle_error == "raise":
                        raise e
                    elif handle_error in ["p1", "p1silent", "nan"]:
                        result[test_name]["statistic"] = np.nan
                        result[test_name]["pvalue"] = 1 if handle_error in ["p1", "p1silent"] else np.nan
                        result[test_name]["error"] = True
                        result[test_name]["data"] = e
                        if handle_error != "p1silent":
                            warnings.warn(f"Error in '{test_name}': {e}")
                    elif isinstance(handle_error, dict):
                        result[test_name]["error"] = True
                        result[test_name] = collections.OrderedDict({**result[test_name], **handle_error})
                    else:
                        raise Exception(f"Unknown error handling strategy: '{handle_error}'")
        
        return test_type, result
    else:
        raise ValueError(f"No test available for data and taget type: {(data_type, target_type)}")

    # TODO: implement sampling
    # def sample_diff(name, difference_function):
    #
    #     split = sklearn.model_selection.StratifiedShuffleSplit(
    #         n_splits=samples_n, train_size=samples_fraction, test_size=1-samples_fraction)
    #     samples = [s for s, _ in split.split(df["values"], df["bin"])]
    #
    #     def single_sample(idx):
    #         df_sample = df.iloc[idx, :]
    #         c0_sample = df_sample[df_sample["bin"] == labels[0]]
    #         c1_sample = df_sample[df_sample["bin"] == labels[1]]
    #         return calc_diff_generic(difference_function, df_sample, labels, c0_sample, c1_sample)
    #
    #     results = [single_sample(idx) for idx in samples]
    #
    #     return {
    #         "statistic": np.mean([ s["statistic"] for s in results]),
    #         "statistic_std": np.std([ s["statistic"] for s in results]),
    #         "pvalue": np.mean([ s["pvalue"] for s in results]),
    #         "pvalue_std": np.std([ s["pvalue"] for s in results]),
    #         "data": {
    #             "samples": results,
    #             "splits": samples
    #         }
    #     }


def calculate_pairwise_comparisons(
        data,
        target,
        labels=None,
        data_types=None,
        target_type=None,
        default_tests_data_target=None,
        return_default_test=True,
        multipletests="bonferroni",
        multipletests_missing_policy="fill",
        nan_policy_infer="raise",
        nan_policy_stats="raise",
        handle_error="raise",
        overall_stats_label="__overall__",
        default_test_label="__default__",
        return_df=True):
    # n, n_notnan, mean, std, median, is_normal

    if labels is None:
        if isinstance(data, pd.DataFrame):
            labels = data.columns.values
        else:
            labels = [f"column_{i}" for i in range(data.shape[1])]

    predefined_default_tests_data_target = collections.OrderedDict([
        (("bin", "bin"), "fisher_exact"),
        (("cat", "bin"), "chi2"),
        (("ord", "bin"), "ranksums"),
        (("num", "bin"), "ranksums"),
        (("num", "ord"), "kendall"),
        (("bin", "num"), "ranksums"),
        (("cat", "num"), "kruskal"),
        (("ord", "num"), "kendall"),
        (("num", "num"), "kendall"),
    ])
    default_tests_data_target = predefined_default_tests_data_target if default_tests_data_target is None \
        else recursive_merge(predefined_default_tests_data_target, default_tests_data_target)

    data = convert_to_numpy(data)

    # init results
    results = collections.OrderedDict([(label, collections.OrderedDict()) for label in labels])

    def get_data_type(label):
        if isinstance(data_types, str) or data_types is None:
            return data_types
        elif isinstance(data_types, Mapping):
            return data_types.get(label, None)
        else:
            raise ValueError(f"Unknown variable type '{data_types}' for label '{label}'")

    # stats
    for i, label in enumerate(labels):
        column_data = data[:, i]
        results[label]["stats"] = collections.OrderedDict([
            (overall_stats_label, calculate_statistics(
                column_data,
                data_type=get_data_type(label),
                nan_policy="omit"))])

    # tests
    for i, label in enumerate(labels):
        column_data = data[:, i]
        (derived_data_type, derived_target_type), test_results = calculate_pairwise_comparison(
            column_data, target,
            data_type=get_data_type(label),
            target_type=target_type,
            handle_error=handle_error,
            nan_policy_stats=nan_policy_stats, nan_policy_infer=nan_policy_infer)
        results[label]["tests"] = test_results
        results[label]["types"] = dict(data=derived_data_type, target=derived_target_type)

        # default tests
        if return_default_test:
            default_test_data_target = default_tests_data_target[(derived_data_type, derived_target_type)]
            results[label]["tests"][default_test_label] = collections.OrderedDict(
                test=default_test_data_target, **test_results[default_test_data_target])

    # multipletests

    if isinstance(multipletests, str):
        multipletests = [multipletests]

    # TODO: handle tests for different types? probably not ... just add column twice with different names and types!
    def extract_pvalues(test_name):
        for column_name, infos in results.items():
            if test_name in infos["tests"]:
                yield column_name, infos["tests"][test_name]["pvalue"]
            else:
                if multipletests_missing_policy == "fill":
                    yield column_name, 1
                elif multipletests_missing_policy == "omit":
                    pass
                else:
                    raise ValueError(f"Unknown multiple tests missing policy: '{multipletests_missing_policy}'")

    # collect tests
    test_names = set()
    for column_name, infos in results.items():
        test_names |= infos["tests"].keys()

    for test_name in test_names:
        extracted_pvalues = list(extract_pvalues(test_name))
        pvalues = [p for _, p in extracted_pvalues]
        column_names = [c for c, _ in extracted_pvalues]
        for method in multipletests:
            _, corrected_pvalues, _, _ = statsmodels.stats.multitest.multipletests(pvalues, method=method)
            for column_name, p in zip(column_names, corrected_pvalues):
                if test_name in results[column_name]["tests"]:
                    results[column_name]["tests"][test_name][f"pvalue___{method}"] = p

    if return_df:
        results = comparisons_to_df(results)

    return results


def comparisons_to_df(differences):
    # TODO: compare with `column_infos_extract_df` which might be a lot faster

    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([(c if isinstance(c, tuple) else (c,)) for c in differences.keys()]),
        columns=pd.MultiIndex(
            levels=[[], [], []],
            codes=[[], [], []],
            names=['type', 'label', 'stat']))

    for c in differences.keys():

        # # add types
        # for label in differences[c]["types"].keys():
        #     column = ("stats", "__overall__", label)
        #     if column not in df:
        #         df.loc[:, column] = np.nan
        #     df.loc[c, column] = differences[c]["types"][label + "_type"]

        # add stats
        for label in differences[c]["stats"].keys():
            for stat in differences[c]["stats"][label].keys():
                column = ("stats", label, stat)
                if column not in df:
                    df.loc[:, column] = np.nan
                df.loc[c, column] = differences[c]["stats"][label][stat]

        # add tests
        for test_name in differences[c]["tests"].keys():
            for key, value in differences[c]["tests"][test_name].items():
                if key not in ["___sampled", "data"]:
                    column = ("tests", test_name, key)
                    if column not in df:
                        df[column] = np.nan
                    df.loc[c, column] = differences[c]["tests"][test_name][key]

            if "___sampled" in differences[c]["tests"][test_name] is not None:
                for key, value in differences[df.column_name[0]]["tests"][test_name]["___sampled"].items():
                    if key != "data":
                        df[("tests", test_name + "___sampled", key)] = \
                            differences[c]["tests"][test_name]["___sampled"][key]

    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df


# if __name__ == '__main__':
#     print("test")
