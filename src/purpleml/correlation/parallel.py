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