import collections
import warnings
from typing import Mapping
from .tests import *
from .utils import calculate_statistics, infer_type
import statsmodels.stats.multitest
import purpleml.utils.misc


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


def calculate_pairwise_comparisons(
        data_matrix,
        target_vector,
        labels=None,
        data_types=None,
        target_type=None,
        default_tests_data_target=None,
        return_default_test=True,
        multipletests="bonferroni",
        multipletests_nan_pvalue_policy=None,
        multipletests_missing_policy="fill",
        nan_policy_infer="raise",
        nan_policy_stats="raise",
        handle_error="raise",
        overall_stats_label="__overall__",
        default_test_label="__default__",
        return_df=True):
    """TODO: finish docs

    Parameters
    ----------
    data : _type_
        _description_
    target : _type_
        _description_
    labels : _type_, optional
        _description_, by default None
    data_types : _type_, optional
        _description_, by default None
    target_type : _type_, optional
        _description_, by default None
    default_tests_data_target : _type_, optional
        _description_, by default None
    return_default_test : bool, optional
        _description_, by default True
    multipletests : str, optional
        _description_, by default "bonferroni"
    multipletests_nan_pvalue_policy : _type_, optional
        What to do with NaN pvalues before applying the correction. `None` will do nothing. "p1" will replace NaN values with a `1`., by default None
    multipletests_missing_policy : str, optional
        What to do when some columns are missing from the test output; "fill" will return a 1 for those columns, "omit" will drop/omit these columns, by default "fill"
    nan_policy_infer : str, optional
        _description_, by default "raise"
    nan_policy_stats : str, optional
        _description_, by default "raise"
    handle_error : str, optional
        _description_, by default "raise"
    overall_stats_label : str, optional
        _description_, by default "__overall__"
    default_test_label : str, optional
        _description_, by default "__default__"
    return_df : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Yields
    ------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
    # n, n_notnan, mean, std, median, is_normal

    if labels is None:
        if isinstance(data_matrix, pd.DataFrame):
            labels = data_matrix.columns.values
        else:
            labels = [f"column_{i}" for i in range(data_matrix.shape[1])]

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
        else purpleml.utils.dicts.recursive_merge(predefined_default_tests_data_target, default_tests_data_target)

    data_matrix = purpleml.utils.misc.convert_to_numpy(data_matrix)

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
        column_data = data_matrix[:, i]
        results[label]["stats"] = collections.OrderedDict([
            (overall_stats_label, calculate_statistics(
                column_data,
                data_type=get_data_type(label),
                nan_policy="omit"))])

    # tests
    for i, label in enumerate(labels):
        column_data = data_matrix[:, i]
        (derived_data_type, derived_target_type), test_results = calculate_pairwise_comparison(
            column_data, target_vector,
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
        if multipletests_nan_pvalue_policy is None:
            pass
        elif multipletests_nan_pvalue_policy == "p1":
            pvalues = [1 if np.isnan(p) else p for p in pvalues]
        else:
            raise ValueError(f"Unknown multiple tests NaN p-value policy: '{multipletests_nan_pvalue_policy}'")


        column_names = [c for c, _ in extracted_pvalues]
        for method in multipletests:
            _, corrected_pvalues, _, _ = statsmodels.stats.multitest.multipletests(pvalues, method=method)
            for column_name, p in zip(column_names, corrected_pvalues):
                if test_name in results[column_name]["tests"]:
                    results[column_name]["tests"][test_name][f"pvalue___{method}"] = p

    if return_df:
        results = comparisons_to_df(results)

    return results


def calculate_pairwise_comparison(
        data_vector,
        target_vector,
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


    tests = predefined_tests if tests is None \
        else purpleml.utils.dicts.recursive_merge(predefined_tests, tests, merge_lists=True)

    if test_kws is None:
        test_kws = {}

    # convert to numpy
    data_vector = purpleml.utils.misc.convert_to_numpy(data_vector)
    target_vector = purpleml.utils.misc.convert_to_numpy(target_vector)

    # collect basic stats
    overlap_n = np.sum(~np.isnan(data_vector) & ~np.isnan(target_vector))

    if infer_type_before_handling_nans:
        if data_type is None:
            data_type = infer_type(data_vector, nan_policy=nan_policy_infer)
        if target_type is None:
            target_type = infer_type(target_vector, nan_policy=nan_policy_infer)

    # handle nans
    data_vector, target_vector = purpleml.utils.misc.handle_nans(data_vector, target_vector, nan_policy=nan_policy_stats)

    if not infer_type_before_handling_nans:
        if data_type is None:
            data_type = infer_type(data_vector, nan_policy=nan_policy_infer)
        if target_type is None:
            target_type = infer_type(target_vector, nan_policy=nan_policy_infer)

    # run tests
    result = collections.OrderedDict()
    test_type = (data_type, target_type)
    if test_type in tests:
        tests = tests[test_type]
        for test_name, test in tests:
            if purpleml.utils.misc.select(test_name, include=test_include, exclude=test_exclude):
                # noinspection PyBroadException
                result[test_name] = collections.OrderedDict(overlap=overlap_n)
                try:
                    s, p, d = test(data_vector, target_vector)
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
        raise ValueError(f"No test available for data and target type: {(data_type, target_type)}")

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
