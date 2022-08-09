
def test_calculate_univariate_comparison():
    import numpy as np
    from purpleml.stats.univariate import calculate_pairwise_comparisons
    
    n, m = 100, 10
    X = np.random.random((n, m))
    y = np.random.random(n)
    
    calculate_pairwise_comparisons(X, y)
