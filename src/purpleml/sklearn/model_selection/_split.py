
import collections
import warnings
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import KFold
import functools
from sklearn.utils.validation import check_array
import numpy as np
from sklearn.model_selection._split import _BaseKFold, check_random_state, type_of_target, column_or_1d

from purpleml.sklearn.model_selection._split_cross_split_search import CrossSplitScoreGridSearchCV


class LenientlyStratifiedKFold(_BaseKFold):
    """Stratified K-Folds cross-validator:
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples in the training sample with slight deviations in the test set if
    there are not ebough samples available.
    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.20
            ``n_splits`` default value will change from 3 to 5 in v0.22.
    shuffle : boolean, optional
        Whether to shuffle each class's samples before splitting into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits(X, y)
    2
    >>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in skf.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        n_samples = y.shape[0]
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts = np.bincount(y_inversed)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            warnings.warn(("All classes have less than n_splits=%d members."
                           "This may randomly result in less splits than anticipated."
                           % self.n_splits), Warning)
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. "
                           "Thus some test splits (n_splits=%d) may not contain all classes."
                           % (min_groups, self.n_splits)), Warning)

        # similar procedure as in `createFolds` of the `caret` library in R
        # (see: https://rdrr.io/rforge/caret/src/R/createFolds.R)
        test_folds = np.zeros(n_samples, dtype=np.int)  # indicates which samples is in which test fold

        # loop through unique y values
        for y_value, count in zip(unique_y, y_counts):
            test_fold_inidices = np.zeros(count, dtype=np.int)
            # the number of times we can assign a sample to every test split
            # (e.g., 3 means that each test split will have at least 3 samples)
            n_sample_assignment_to_all_splits = int(count / self.n_splits)
            # calculate the number of samples covered by equally assigned samples (may be 0)
            n_samples_in_complete_splits = self.n_splits * n_sample_assignment_to_all_splits
            # create a vector of integers from `1:n_splits` as many times as possible
            # Note that if the number of samples in a class is less than `n_splits`, nothing is produced here.
            test_fold_inidices[:n_samples_in_complete_splits] = \
                np.repeat(np.arange(self.n_splits), n_sample_assignment_to_all_splits)
            # add enough random integers to distribute the rest of the samples
            test_fold_inidices[n_samples_in_complete_splits:] = \
                sorted(np.random.choice(np.arange(self.n_splits), count - n_samples_in_complete_splits, replace=False))
            # shuffle if requested
            if self.shuffle:
                rng.shuffle(test_fold_inidices)
            test_folds[y == y_value] = test_fold_inidices

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            if any(test_folds == i):
                yield test_folds == i

    # def get_n_splits(self, X=None, y=None, groups=None):
    #     """
    #     Returns the maximum (!) number of splits.
    #     Due to randomness the number of spits may differ.
    #
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         Training data, where n_samples is the number of samples
    #         and n_features is the number of features.
    #         Note that providing ``y`` is sufficient to generate the splits and
    #         hence ``np.zeros(n_samples)`` may be used as a placeholder for
    #         ``X`` instead of actual training data.
    #     y : array-like, shape (n_samples,)
    #         The target variable for supervised learning problems.
    #         Stratification is done based on the y labels.
    #     groups : object
    #         Always ignored, exists for compatibility.
    #
    #     Returns
    #     -------
    #     The maximum number of splits
    #
    #     """
    #     return super().get_n_splits(X, y, groups)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


class GroupSplittingStrategy:

    def __init__(self, splitting_strategy, check_group_label_homogeneity=True):
        self.splitting_strategy = splitting_strategy
        self.check_group_label_homogeneity = check_group_label_homogeneity

    def split(self, X, y=None, groups=None):

        if X is not None:
            X = check_array(X, ensure_2d=True, dtype=None, force_all_finite="allow-nan")

        if y is not None:
            y = check_array(y, ensure_2d=False, dtype=None)

        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        # boolean selection vector to mark the first occurrence of each group
        select_one = np.zeros(len(groups), dtype=bool)

        # will hold boolean selection vectors for each group
        select_group = {}

        # fill the index structures above
        for i, g in enumerate(groups):
            if g not in select_group:
                select_group[g] = np.zeros(len(groups), dtype=bool)
                select_one[i] = True
            select_group[g][i] = True

        # check group labels
        if self.check_group_label_homogeneity:
            for g, idx in select_group.items():
                if len(np.unique(y[idx])) > 1:
                    raise Exception(
                        "All groups must have the same label for stratification to be valid. "
                        + "Group '{}' has the following labels: {}".format(
                            g, y[idx]))

        # split first occurrences without considering X
        X_split = X[select_one, :]
        y_split = y[select_one] if y is not None else None
        split = self.splitting_strategy.split(X=X_split, y=y_split)

        # unique group labels
        group_list = groups[select_one]
        # helper variable to convert boolean selection vectors to indices
        index = np.arange(len(groups))
        # yield splits
        for train_index, test_index in split:
            yield \
                index[functools.reduce(np.logical_or, [select_group[g] for g in group_list[train_index]])], \
                index[functools.reduce(np.logical_or, [select_group[g] for g in group_list[test_index]])]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitting_strategy.get_n_splits(X, y, groups)


class StratifiedGroupKFold(GroupSplittingStrategy):

    def __init__(self, *args, check_group_label_homogeneity=True, **kwargs):
        super().__init__(StratifiedKFold(*args, **kwargs), check_group_label_homogeneity)


class LenientlyStratifiedGroupKFold(GroupSplittingStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(LenientlyStratifiedKFold(*args, **kwargs))


class StratifiedGroupShuffleSplit(GroupSplittingStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(StratifiedShuffleSplit(*args, **kwargs))


class ShuffledGroupKFold(GroupSplittingStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(splitting_strategy=KFold(*args, **kwargs), check_group_label_homogeneity=False)


class CheatSplit:

    def __init__(self, splitting_strategy=None):
        self.splitting_strategy = splitting_strategy

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitting_strategy.split(X=X, y=y, group=groups)

    def split(self, X, y=None, groups=None):
        splits = self.splitting_strategy.split(X=X, y=y, groups=groups)
        for train_idx, test_idx in splits:
            yield np.concatenate([train_idx, test_idx]), test_idx


class AllSplit:

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        yield np.arange(X.shape[0]), np.arange(X.shape[0])


class Undersampling:
    """
    First under samples data before applying splitting strategy.
    """

    def __init__(self, splitting_strategy, replace=False, random_state=None):
        self.splitting_strategy = splitting_strategy
        self.replace = replace
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        # TODO: check whether we should actually undersample before calling get_n_splits?
        return self.splitting_strategy.get_n_splits(X=X, y=y, group=groups)

    def split(self, X, y=None, groups=None):

        n_classes = len(np.unique(y))
        if n_classes > 2:
            raise ValueError(f"More than two classes currently not supported. Given: {n_classes}")

        # init random state
        rs = np.random.RandomState(self.random_state)

        # count classes
        y_counts = collections.Counter(y).most_common(2)

        idx_c0 = np.arange(len(y))[y == y_counts[1][0]]
        idx_c1 = rs.choice(np.arange(len(y))[y == y_counts[0][0]], y_counts[1][1], replace=self.replace)

        idx = np.concatenate([idx_c0, idx_c1])

        X_sampled = X[idx, :]
        y_sampled = y[idx]
        if groups is not None:
            groups_sampled = groups[idx]
        else:
            groups_sampled = None

        splits = self.splitting_strategy.split(X=X_sampled, y=y_sampled, groups=groups_sampled)
        for train_idx, test_idx in splits:
            yield idx[train_idx], idx[test_idx]


class Oversampling:
    """
    Only over samples training.
    """

    def __init__(self, splitting_strategy, shuffle=True, random_state=None):
        self.splitting_strategy = splitting_strategy
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.splitting_strategy.split(X=X, y=y, group=groups)

    def split(self, X, y=None, groups=None):

        # init random state
        rs = np.random.RandomState(self.random_state)

        splits = self.splitting_strategy.split(X=X, y=y, groups=groups)
        for train_idx, test_idx in splits:

            y_train = y[train_idx]

            counts = collections.Counter(y_train).most_common()
            max_count = counts[0][1]

            idx_train_oversampled = []
            for element, count in collections.Counter(y_train).most_common():
                idx_element = train_idx[y_train == element]
                n_repeat = max_count // len(idx_element)
                idx_oversampled = np.repeat(idx_element, n_repeat)
                idx_oversampled = np.concatenate([
                    idx_oversampled,
                    rs.choice(idx_element, size=max_count - n_repeat * len(idx_element), replace=False)])
                idx_train_oversampled.append(idx_oversampled)
            idx_train_oversampled = np.concatenate(idx_train_oversampled)

            if self.shuffle:
                rs.shuffle(idx_train_oversampled)

            yield idx_train_oversampled, test_idx

