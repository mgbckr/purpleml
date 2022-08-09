from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt


def subplots(n, n_per_break, break_mode="rows", figsize_single=None, **kwargs):
    """Create axes array with `n_per_break` columns given an overall number of plots of `n`.
    

    Parameters
    ----------
    n : int
        number of plots
    n_per_break : int
        plots until break
    break_mode : str, optional
        where to break; can be "rows" or "columns", by default "rows"
    figsize_single : tuple, optional
        figure size for one individual plot, by default None
    kwargs : dict, optional
        kwargs are passed to `matplotlib.pyplot.subplots`


    Returns
    -------
    tuple
        result of `matplotlib.pyplot.subplots`: fig, axes

    Raises
    ------
    ValueError
        If wrong `break_mode` is given.
    """
    
    n_breaks = int(np.ceil(n / n_per_break))
    n_per_break = min(n_per_break, n)

    if figsize_single is not None:
        if isinstance(figsize_single , int):
            kwargs["figsize"] = (kwargs["figsize"] * n_per_break, kwargs["figsize"] * n_breaks)
        else:
            kwargs["figsize"] = (figsize_single[0] * n_per_break, figsize_single[1] * n_breaks)

    if break_mode == "row":
        return plt.subplots(n_breaks, n_per_break, **kwargs)
    elif break_mode == "columns":
        return plt.subplots(n_per_break, n_breaks, **kwargs)
    else:
        raise ValueError(f"Unknown break mode: '{break_mode}'.")
