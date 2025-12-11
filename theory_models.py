"""
Theory predictions for inflation models in the r-ns plane.
Contains functions for polynomial potentials, alpha-unity models, and plot elements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_style import alpha_unity_style_dict


# ============================================================================
# Helper Functions for r-ns Relations
# Taken from BK18 plotting script at http://bicepkeck.org/bk18_2021_release.html
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
    return (r - 16) / (8.0 * ns - 8 + r) / 2.0


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
    return 8.0 * (1 - ns) * p / (2.0 + p)


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
    return (4 * N - p - 4) / (4.0 * N + p)


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

# Model parameters: N* values for each model
# These can be single values or [N_min, N_max] ranges
# Publication values: Starobinsky N*=51, Higgs N*=55
ALPHA_UNITY_MODELS = {
    "Starobinsky $R^2$": 51,  # Single value for publication
    "Higgs": 55,  # Single value for publication
}


def get_alpha_unity_model_prediction(model_name):
    """
    Get r and ns predictions for alpha-unity model.

    For models with N* ranges, returns arrays of predictions across the range.
    For single N* values, returns single (ns, r) tuple.

    Parameters
    ----------
    model_name : str
        Name of the model (must be in ALPHA_UNITY_MODELS)

    Returns
    -------
    tuple
        (ns, r) predictions for the model
        - If N* is a single value: (float, float)
        - If N* is a range: (array, array)
    """
    if model_name not in ALPHA_UNITY_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(ALPHA_UNITY_MODELS.keys())}"
        )

    N = np.atleast_1d(ALPHA_UNITY_MODELS[model_name])
    r = 12 / N**2
    ns = 1 - 2 / N

    # Return scalars if single value, arrays if range
    if len(N) == 1:
        return float(ns[0]), float(r[0])
    return ns, r


def get_alpha_unity_label(model_name):
    """
    Get the legend label for an alpha-unity model including N* information.

    Parameters
    ----------
    model_name : str
        Name of the model (must be in ALPHA_UNITY_MODELS)

    Returns
    -------
    str
        Label with N* value(s) in parentheses
    """
    if model_name not in ALPHA_UNITY_MODELS:
        return model_name

    N_star = np.atleast_1d(ALPHA_UNITY_MODELS[model_name])
    if len(N_star) > 1:
        return f"{model_name} (${N_star[0]}\\leq N_\\star\\leq {N_star[1]}$)"
    else:
        return f"{model_name} ($N_\\star={N_star[0]}$)"


# ============================================================================
# Polynomial Alpha-Attractor Functions
# ============================================================================


def load_polynomial_alpha_attractor_line(
    N_star, k=2, data_dir="polynomial_alpha_attractor_lines"
):
    """
    Load precomputed polynomial alpha-attractor line from file.

    Parameters
    ----------
    N_star : int
        Number of e-folds
    k : int
        Power in the potential (default: 2)
    data_dir : str
        Directory containing the data files

    Returns
    -------
    tuple
        (ns_array, r_array) for the alpha-attractor line
    """
    filename = f"polynomial_alpha_attractor_k_{k}_Nstar_{N_star}.txt"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Polynomial alpha-attractor line file not found: {filepath}"
        )

    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]  # ns, r


def get_polynomial_alpha_attractor_label(N_range, k=2):
    """
    Get the legend label for polynomial alpha-attractor.

    Parameters
    ----------
    N_range : tuple
        (N_min, N_max) e-fold range
    k : int
        Power in the potential

    Returns
    -------
    str
        Formatted label for legend
    """
    return f"Polynomial $\\alpha$-attractor\n($k={k},\\, {N_range[0]}\\leq\\! N_\\star\\!\\leq {N_range[1]}$)"


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

    default_kwargs = {"ls": "-", "color": "k", "lw": 1, "alpha": 0.8, "zorder": -5}
    default_kwargs.update(kwargs)

    ns = np.arange(ns_range[0], ns_range[1], 0.0001)
    line = ax.plot(ns, r_ns(ns, 1), **default_kwargs)
    return line[0]


def add_concave_convex_labels(
    ax, ns_center=0.995, offset_distance_display=20, fontsize=10
):
    """
    Add 'Concave' and 'Convex' text labels to plot with dynamic positioning.

    The labels are positioned perpendicular to the p=1 dividing line, with
    rotation matching the line angle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add labels to
    ns_center : float
        Central ns value along the p=1 line for label placement
    offset_distance_display : float
        Distance from line in display coordinates (pixels)
    fontsize : int
        Font size for labels
    """
    # Calculate center point on p=1 line
    r_center = r_ns(ns_center, 1)

    # Calculate the angle of the line at this position
    ns_line = np.linspace(ns_center - 0.01, ns_center + 0.01, 2)
    r_line = r_ns(ns_line, 1)
    line_display = ax.transData.transform(np.column_stack([ns_line, r_line]))
    dx = line_display[1, 0] - line_display[0, 0]
    dy = line_display[1, 1] - line_display[0, 1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate perpendicular direction (90 degrees to the line)
    perp_angle = angle + 90

    # Calculate offsets in display coordinates
    dx_perp_display = offset_distance_display * np.cos(np.radians(perp_angle))
    dy_perp_display = offset_distance_display * np.sin(np.radians(perp_angle))

    # Transform center point to display coordinates
    center_display = ax.transData.transform([[ns_center, r_center]])[0]

    # Add offsets in display coordinates
    concave_display = [
        center_display[0] - dx_perp_display,
        center_display[1] - dy_perp_display,
    ]
    convex_display = [
        center_display[0] + dx_perp_display,
        center_display[1] + dy_perp_display,
    ]

    # Transform back to data coordinates
    concave_data = ax.transData.inverted().transform(concave_display)
    convex_data = ax.transData.inverted().transform(convex_display)

    # Place text at calculated positions
    ax.text(
        concave_data[0],
        concave_data[1],
        "Concave",
        rotation=angle,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    ax.text(
        convex_data[0],
        convex_data[1],
        "Convex",
        rotation=angle,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def add_monomial_potentials(
    ax,
    p_values=[1 / 3, 2 / 3, 1],
    N_range=(47, 57),
    add_labels=True,
    label_offsets=None,
    return_handles=False,
    **kwargs,
):
    """
    Add monomial potential curves V ∝ φ^p to plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    p_values : list
        List of power values to plot
    N_range : tuple
        (N_min, N_max) e-fold range for each curve (default: 47, 57)
    add_labels : bool
        Whether to add φ^p text labels
    label_offsets : dict or None
        Dictionary mapping p_label strings to [dx, dy] offsets.
        If None, uses defaults optimized for linear scale
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
    default_kwargs = {"ls": "-", "color": "r", "lw": 1.2, "alpha": 1, "zorder": -1}
    default_kwargs.update(kwargs)

    # Default label offsets (optimized for linear scale)
    if label_offsets is None:
        label_offsets = {
            "1/3": [0.00025, 0.0025],
            "2/3": [0.00025, 0.0025],
            "": [0.00025, 0.0025],
        }

    p_labels = {1 / 3: "1/3", 2 / 3: "2/3", 1: ""}
    legend_label = (
        r"$V(\phi) \propto \phi^{n},\, n=1, \sfrac{2}{3}, \sfrac{1}{3}$"
        + f"\n(${N_range[0]}\\leq\\! N_\\star\\!\\leq {N_range[1]}$)"
    )
    line_handle = None

    for p in p_values:
        p_label = p_labels.get(p, str(p))
        lbl = None  # Don't add label directly, use custom handler

        ns = np.arange(ns_N(N_range[0], p), ns_N(N_range[1], p), 0.0001)
        line = ax.plot(ns, r_ns(ns, p), label=lbl, **default_kwargs)

        if p == 1:
            line_handle = line[0]

        # Add text labels if requested
        if add_labels and p_label in label_offsets:
            ns_end = ns_N(N_range[1], p)
            r_end = r_ns(ns_end, p)
            dx, dy = label_offsets[p_label]
            ax.text(
                ns_end + dx,
                r_end + dy,
                f"$\\phi^{{{p_label}}}$",
                color=default_kwargs["color"],
                ha="left",
            )

    if return_handles:
        return line_handle, legend_label
    return line_handle


def add_efold_shading_monomial(ax, N_range=(47, 57), ns_range=(0.96, 1.0), **kwargs):
    """
    Add shaded region showing e-fold range for monomial potentials V ∝ φⁿ.

    This shading is only mathematically valid for monomial (power-law) potentials.
    For other potential shapes, the N* constraints may differ.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    N_range : tuple
        (N_min, N_max) e-fold range (default: 47, 57)
    ns_range : tuple
        (ns_min, ns_max) range for shading
    **kwargs : dict
        Additional arguments passed to ax.fill_between()

    Returns
    -------
    PolyCollection
        The shaded region
    """
    default_kwargs = {"lw": 1, "facecolor": "0.8", "edgecolor": "0.6", "zorder": -10}
    default_kwargs.update(kwargs)

    ns = np.linspace(ns_range[0], ns_range[1], 100)
    shading = ax.fill_between(
        ns,
        r_from_N_star_ns(N_range[0], ns),
        r_from_N_star_ns(N_range[1], ns),
        **default_kwargs,
    )
    return shading


def add_alpha_unity_model_markers(ax, models=None, return_handles=False):
    """
    Add markers for specific inflation models (Starobinsky, Higgs, etc.).

    Supports both single N* values (displayed as scatter points) and
    N* ranges (displayed as markers at min/max with connecting line).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    models : list or None
        List of model names to plot. If None, plots all available models
    return_handles : bool
        Whether to return legend handles and labels (for advanced usage)

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
        if model_name not in ALPHA_UNITY_MODELS:
            continue
        if model_name not in alpha_unity_style_dict:
            continue

        # Get N* from theory_models (single source of truth)
        N_star = np.atleast_1d(ALPHA_UNITY_MODELS[model_name])

        # Get styling from plot_style
        style_dict = alpha_unity_style_dict[model_name]

        # Get label with N* info
        label = get_alpha_unity_label(model_name)

        if len(N_star) == 2:
            # Range: plot markers at min and max only, with connecting line
            N_min, N_max = N_star[0], N_star[1]

            # Calculate r and ns for alpha=1 models: r = 12/N^2, ns = 1 - 2/N
            ns_values = 1 - 2 / np.array([N_min, N_max])
            r_values = 12 / np.array([N_min, N_max]) ** 2

            # Plot connecting line
            line = ax.plot(
                ns_values,
                r_values,
                color="k",
                lw=style_dict["lw"],
                zorder=999998,
            )[0]

            # Plot markers at endpoints
            scatter = ax.scatter(
                ns_values,
                r_values,
                marker=style_dict["marker"],
                s=style_dict["s"],
                c=style_dict["color"],
                edgecolors=style_dict["edgecolor"],
                linewidths=1.2,
                label=label,
                zorder=999999,
            )

            if return_handles:
                handles.append(scatter)
                labels.append(label)

        elif len(N_star) > 2:
            # Multiple values: plot all with line
            this_r = 12 / N_star**2
            this_ns = 1 - 2 / N_star

            line = ax.plot(
                this_ns,
                this_r,
                color=style_dict["color"],
                lw=style_dict["lw"],
                marker=style_dict["marker"],
                ms=style_dict["ms"],
                label=label,
            )[0]

            if return_handles:
                handles.append(line)
                labels.append(label)
        else:
            # Single point: plot as scatter
            this_r = 12 / N_star**2
            this_ns = 1 - 2 / N_star

            scatter_kwargs = {
                k: v for k, v in style_dict.items() if k not in ["lw", "ms"]
            }
            if "s" not in scatter_kwargs:
                scatter_kwargs["s"] = style_dict.get("ms", 6) ** 2

            handle = ax.scatter(
                this_ns, this_r, **scatter_kwargs, label=label, zorder=999999
            )

            if return_handles:
                handles.append(handle)
                labels.append(label)

    if return_handles:
        return handles, labels
    return None


def get_polynomial_alpha_attractor_label(N_star, k=2):
    """
    Get the legend label for polynomial alpha-attractor.

    Parameters
    ----------
    N_star : int or array-like
        Single N* value or iterable of N* values
    k : int
        Power in the potential

    Returns
    -------
    str
        Formatted label for legend
    """
    N_star = np.atleast_1d(N_star)

    if len(N_star) == 1:
        # Single value: show exact N*
        return f"Polynomial $\\alpha$-attractor\n($k={k},\\, N_\\star={N_star[0]}$)"
    elif len(N_star) == 2:
        # Two values: show range
        N_min, N_max = min(N_star), max(N_star)
        return f"Polynomial $\\alpha$-attractor\n($k={k},\\, {N_min}\\leq\\! N_\\star\\!\\leq {N_max}$)"
    else:
        # More than 2: just show model name (single line)
        return f"Polynomial $\\alpha$-attractor ($k={k}$)"


def add_polynomial_alpha_attractor(
    ax,
    N_star=None,
    k=2,
    return_handles=False,
    data_dir="polynomial_alpha_attractor_lines",
    **kwargs,
):
    """
    Add polynomial alpha-attractor lines to plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    N_star : int or array-like, optional
        Single N* value or iterable of N* values. Default is [47, 57]
    k : int
        Power in the potential (default: 2)
    return_handles : bool
        Whether to return legend handle and label
    data_dir : str
        Directory containing the precomputed data files
    **kwargs : dict
        Additional arguments passed to ax.plot()

    Returns
    -------
    tuple or None
        If return_handles=True: (handle, label) for legend
        If return_handles=False: None
    """
    # Handle default and convert to array
    if N_star is None:
        N_star = [47, 57]
    N_star = np.atleast_1d(N_star)

    default_kwargs = {"color": "k", "ls": ":", "lw": 1.0}
    default_kwargs.update(kwargs)

    line_handle = None

    for N in N_star:
        try:
            ns, r = load_polynomial_alpha_attractor_line(int(N), k=k, data_dir=data_dir)
            line = ax.plot(ns, r, **default_kwargs)
            if line_handle is None:
                line_handle = line[0]
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    label = get_polynomial_alpha_attractor_label(N_star, k)

    if return_handles:
        return line_handle, label
    return None
