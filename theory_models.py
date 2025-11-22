"""
Theory predictions for inflation models in the r-ns plane.
Contains functions for polynomial potentials, alpha-unity models, and plot elements.
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_style import alpha_unity_style_dict


# ============================================================================
# Helper Functions for r-ns Relations
# # Taken from BK18 plotting script at http://bicepkeck.org/bk18_2021_release.html
# ============================================================================

def N_r_ns(r, ns):
    """
    Calculate number of e-folds from r and ns.
    
    Parameters
    ----------
    r : float or array
        Tensor-to-scalar ratio
    ns : float or array
        Scalar spectral index
        
    Returns
    -------
    float or array
        Number of e-folds N
    """
    return (r - 16) / (8. * ns - 8 + r) / 2.


def r_ns(ns, p):
    """
    Calculate r as function of ns for power-law potential V ∝ φ^p.
    
    Parameters
    ----------
    ns : float or array
        Scalar spectral index
    p : float
        Power of the potential
        
    Returns
    -------
    float or array
        Tensor-to-scalar ratio r
    """
    return 8. * (1 - ns) * p / (2. + p)


def ns_N(N, p):
    """
    Calculate ns as function of N for power-law potential (first order).
    
    Parameters
    ----------
    N : float or array
        Number of e-folds
    p : float
        Power of the potential
        
    Returns
    -------
    float or array
        Scalar spectral index ns
    """
    return (4 * N - p - 4) / (4. * N + p)


def r_from_N_star_ns(N_star, ns):
    """
    Calculate r from N_star and ns for polynomial potentials.
    
    Parameters
    ----------
    N_star : float or array
        Number of e-folds
    ns : float or array
        Scalar spectral index
        
    Returns
    -------
    float or array
        Tensor-to-scalar ratio r
    """
    denom = 1 - 2 * N_star
    num = 2 * N_star * (8 * ns - 8) + 16
    return num / denom


# ============================================================================
# Alpha-Unity Model Definitions
# ============================================================================

# Model parameters: {model_name: N_efolds}
ALPHA_UNITY_MODELS = {
    "Starobinsky $R^2$": 50,
    "Higgs": 57
}


def get_alpha_unity_model_prediction(model_name):
    """
    Get r and ns predictions for alpha-unity model.
    
    Parameters
    ----------
    model_name : str
        Name of the model (must be in ALPHA_UNITY_MODELS)
        
    Returns
    -------
    tuple
        (ns, r) predictions for the model
    """
    if model_name not in ALPHA_UNITY_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ALPHA_UNITY_MODELS.keys())}")
    
    N = ALPHA_UNITY_MODELS[model_name]
    r = 12 / N**2
    ns = 1 - 2 / N
    return ns, r


# ============================================================================
# Functions to Add Theory Elements to Plots
# ============================================================================

def add_concave_convex_divide(ax, ns_range=None, **kwargs):
    """
    Add the concave/convex divide line (p=1 power law).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    ns_range : tuple or None
        (ns_min, ns_max) for the line. If None, uses (0.9, 1.1)
    **kwargs : dict
        Additional arguments passed to ax.plot()
        
    Returns
    -------
    matplotlib.lines.Line2D
        The plotted line
    """
    if ns_range is None:
        ns_range = (0.9, 1.1)
    
    default_kwargs = {'ls': '-', 'color': 'k', 'lw': 1, 'alpha': 0.8}
    default_kwargs.update(kwargs)
    
    ns = np.arange(ns_range[0], ns_range[1], 0.0001)
    line = ax.plot(ns, r_ns(ns, 1), **default_kwargs)
    return line[0]


def add_concave_convex_labels(ax, concave_pos=(0.875, 0.075), convex_pos=(0.9, 0.175),
                               rotation=-25, fontsize=10):
    """
    Add 'Concave' and 'Convex' text labels to plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add labels to
    concave_pos : tuple
        (x, y) position in axes coordinates for 'Concave' label
    convex_pos : tuple
        (x, y) position in axes coordinates for 'Convex' label
    rotation : float
        Text rotation angle in degrees
    fontsize : int
        Font size for labels
    """
    ax.text(*concave_pos, "Concave",
            transform=ax.transAxes, ha="center", va="center",
            rotation=rotation, fontsize=fontsize)
    
    ax.text(*convex_pos, "Convex",
            transform=ax.transAxes, ha="center", va="center",
            rotation=rotation, fontsize=fontsize)


def add_polynomial_potentials(ax, p_values=[1/3, 2/3, 1], N_range=(50, 60),
                               add_labels=True, label_offsets=None,
                               return_handles=False, **kwargs):
    """
    Add polynomial potential curves V ∝ φ^p to plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    p_values : list
        List of power values to plot
    N_range : tuple
        (N_min, N_max) e-fold range for each curve
    add_labels : bool
        Whether to add φ^p text labels
    label_offsets : dict or None
        Dictionary mapping p_label strings to [dx, dy] offsets.
        If None, uses defaults optimized for log scale
    return_handles : bool
        Whether to return legend handle and label
    **kwargs : dict
        Additional arguments passed to ax.plot()
        
    Returns
    -------
    Line2D or tuple
        If return_handles=False: the p=1 line handle
        If return_handles=True: (line_handle, legend_label) for p=1
    """
    default_kwargs = {'ls': '-', 'color': 'r', 'lw': 1.2, 'alpha': 1}
    default_kwargs.update(kwargs)
    
    # Default label offsets (optimized for log scale)
    if label_offsets is None:
        label_offsets = {
            "1/3": [0.00025, 0.0025],
            "2/3": [4*0.00025, -2.5*0.0025],
            "": [0.00025, 2*0.0025]
        }
    
    p_labels = {1/3: "1/3", 2/3: "2/3", 1: ""}
    legend_label = r"$V(\phi) \propto \phi^{n},\, n=1, \sfrac{2}{3}, \sfrac{1}{3}$"
    line_handle = None
    
    for p in p_values:
        p_label = p_labels.get(p, str(p))
        lbl = legend_label if p == 1 else None
        
        ns = np.arange(ns_N(N_range[0], p), ns_N(N_range[1], p), 0.0001)
        line = ax.plot(ns, r_ns(ns, p), label=lbl, **default_kwargs)
        
        if p == 1:
            line_handle = line[0]
        
        # Add text labels if requested
        if add_labels and p_label in label_offsets:
            ns_end = ns_N(N_range[1], p)
            r_end = r_ns(ns_end, p)
            dx, dy = label_offsets[p_label]
            ax.text(ns_end + dx, r_end + dy, f"$\phi^{{{p_label}}}$",
                   color=default_kwargs['color'], ha="left")
    
    if return_handles:
        return line_handle, legend_label
    return line_handle


def add_efold_shading(ax, N_range=(50, 60), ns_range=(0.96, 1.0), **kwargs):
    """
    Add shaded region showing e-fold range for polynomial potentials.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    N_range : tuple
        (N_min, N_max) e-fold range
    ns_range : tuple
        (ns_min, ns_max) range for shading
    **kwargs : dict
        Additional arguments passed to ax.fill_between()
        
    Returns
    -------
    PolyCollection
        The shaded region
    """
    default_kwargs = {
        'lw': 1,
        'facecolor': "0.8",
        'edgecolor': "0.6",
        'zorder': -1
    }
    default_kwargs.update(kwargs)
    
    ns = np.linspace(ns_range[0], ns_range[1], 100)
    shading = ax.fill_between(ns,
                               r_from_N_star_ns(N_range[0], ns),
                               r_from_N_star_ns(N_range[1], ns),
                               **default_kwargs)
    return shading


def add_alpha_unity_model_markers(ax, models=None, return_handles=True):
    """
    Add markers for specific inflation models (Starobinsky, Higgs, etc.).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    models : list or None
        List of model names to plot. If None, plots all available models
    return_handles : bool
        Whether to return legend handles and labels
        
    Returns
    -------
    tuple or None
        If return_handles=True: (handles, labels) lists for legend
        If return_handles=False: None
    """
    if models is None:
        models = list(ALPHA_UNITY_MODELS.keys())
    
    handles = []
    labels = []
    
    for model_name in models:
        ns, r = get_alpha_unity_model_prediction(model_name)
        handle = ax.scatter(ns, r, **alpha_unity_style_dict[model_name],
                          label=model_name)
        if return_handles:
            handles.append(handle)
            labels.append(model_name)
    
    if return_handles:
        return handles, labels
    return None
