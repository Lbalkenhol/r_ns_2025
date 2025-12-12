"""
Utilities for handling legends with GetDist integration.
"""

import matplotlib.pyplot as plt
from plot_style import (
    style_dict,
    HandlerTwoLines,
    HandlerLineWithEndMarkers,
    HandlerMonomial,
    HandlerLineMultiline,
    HandlerScatterMultiline,
    alpha_unity_style_dict,
)
from theory_models import ALPHA_UNITY_MODELS, get_alpha_unity_label
import numpy as np


def create_dummy_plot_elements_for_legend(all_dat, return_entries=False):
    """
    Create dummy plot elements for legend that match GetDist contours.

    This function creates invisible plot elements that can be picked up by
    matplotlib's legend system, matching the style of GetDist contour plots.

    Parameters
    ----------
    all_dat : list
        List of dataset keys (must be in style_dict)
    return_entries : bool
        If True, returns handles, labels, and handler_map for manual legend creation
        If False, just creates the dummy elements in the plot

    Returns
    -------
    tuple or None
        If return_entries=True: (handles, labels, handler_map)
        If return_entries=False: None

    Notes
    -----
    Special handling for forecast datasets:
    - If both "FC" and "FC_DESI" are in all_dat, they are combined into a
      single two-line legend entry labeled "CMB 2030s"
    """
    ax = plt.gca()
    legend_handles = []
    legend_labels = []
    handler_map = {}

    # Check if we need the special two-line forecast entry
    has_forecasts = "FC" in all_dat and "FC_DESI" in all_dat

    for dat in all_dat:
        if dat not in style_dict:
            continue

        # Special handling for forecast combination
        if has_forecasts:
            if dat == "FC":
                # Add the custom two-line entry
                dummy_handle_combined = plt.Line2D([], [], color="k")
                legend_handles.append(dummy_handle_combined)
                legend_labels.append("CMB 2030s")
                handler_map[dummy_handle_combined] = HandlerTwoLines()
                continue
            elif dat == "FC_DESI":
                # Skip - already handled with FC
                continue

        # Regular entries
        if style_dict[dat]["filled"]:
            handle = ax.fill_between(
                [2, 2],
                [2, 2],
                [2, 2],
                color=style_dict[dat]["colour"],
                lw=0,
                label=style_dict[dat]["label"],
            )
        else:
            handle = ax.plot(
                [2, 2],
                [2, 2],
                color=style_dict[dat]["colour"],
                ls=style_dict[dat]["ls"],
                label=style_dict[dat]["label"],
            )[0]

        legend_handles.append(handle)
        legend_labels.append(style_dict[dat]["label"])

    if return_entries:
        return legend_handles, legend_labels, handler_map
    return None


def add_monomial_legend_entry(
    legend_handles,
    legend_labels,
    handler_map,
    N_range=(47, 57),
    yoffset=0.0,
    use_sfrac=True,
):
    """
    Add a monomial potential legend entry with custom handler.

    Parameters
    ----------
    legend_handles : list
        List of legend handles to append to
    legend_labels : list
        List of legend labels to append to
    handler_map : dict
        Handler map to update
    N_range : tuple
        (N_min, N_max) e-fold range for label
    yoffset : float
        Vertical offset for multiline entries
    use_sfrac : bool
        Whether to use \\sfrac for fractions (nicer but can fail on some servers)
    """
    if use_sfrac:
        label = (
            r"$V(\phi) \propto \phi^{n},\, n=1, \sfrac{2}{3}, \sfrac{1}{3}$"
            + f"\n(${N_range[0]}\\leq\\! N_\\star\\!\\leq {N_range[1]}$)"
        )
    else:
        label = (
            r"$V(\phi) \propto \phi^{n},\, n=1, \frac{2}{3}, \frac{1}{3}$"
            + f"\n(${N_range[0]}\\leq\\! N_\\star\\!\\leq {N_range[1]}$)"
        )

    dummy_handle = plt.Line2D([], [], color="k")
    legend_handles.append(dummy_handle)
    legend_labels.append(label)
    handler_map[dummy_handle] = HandlerMonomial(
        facecolor="0.8",
        edgecolor="0.6",
        linecolor="r",
        linewidth=0.8,
        yoffset=yoffset,
    )


def add_alpha_attractor_legend_entry(
    legend_handles, legend_labels, handler_map, N_star=None, k=2, yoffset=0.0
):
    """
    Add a polynomial alpha-attractor legend entry with custom handler.

    Parameters
    ----------
    legend_handles : list
        List of legend handles to append to
    legend_labels : list
        List of legend labels to append to
    handler_map : dict
        Handler map to update
    N_star : int or array-like, optional
        Single N* value or iterable of N* values. Default is [47, 57]
    k : int
        Power in the potential
    yoffset : float
        Vertical offset for multiline entries (auto-set to 0 for >2 N* values)
    """
    from theory_models import get_polynomial_alpha_attractor_label

    # Handle default and convert to array
    if N_star is None:
        N_star = [47, 57]
    N_star = np.atleast_1d(N_star)

    label = get_polynomial_alpha_attractor_label(N_star, k)

    # For more than 2 N* values, label is single line so no yoffset needed
    if len(N_star) > 2:
        yoffset = 0.0

    dummy_handle = plt.Line2D([], [], color="k")
    legend_handles.append(dummy_handle)
    legend_labels.append(label)
    handler_map[dummy_handle] = HandlerLineMultiline(
        linestyle=":",
        linewidth=1.0,
        color="k",
        yoffset=yoffset,
    )


def add_model_handlers_to_legend(
    ax, handles=None, labels=None, handler_map=None, **legend_kwargs
):
    """
    Add legend with automatic handling of special model entries (Starobinsky, etc.).

    This function gets all handles and labels from the current axes (or uses provided ones),
    identifies models that need special handlers (like Starobinsky R² with N* range),
    and creates a legend with the appropriate custom handlers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add legend to
    handles : list, optional
        List of legend handles. If None, gets from ax.get_legend_handles_labels()
    labels : list, optional
        List of legend labels. If None, gets from ax.get_legend_handles_labels()
    handler_map : dict, optional
        Existing handler_map to extend (e.g., from forecast handling)
    **legend_kwargs : dict
        Additional keyword arguments passed to ax.legend()

    Returns
    -------
    matplotlib.legend.Legend
        The created legend object
    """
    # Get handles and labels if not provided
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()

    # Create or extend handler_map for models with special requirements
    if handler_map is None:
        handler_map = {}

    # Check each label and add appropriate handler if needed
    for handle, label in zip(handles, labels):
        # Check if this is an alpha-unity model with N* range (2 values)
        for model_name in ALPHA_UNITY_MODELS:
            model_label = get_alpha_unity_label(model_name)
            if label == model_label and model_name in alpha_unity_style_dict:
                N_star = np.atleast_1d(ALPHA_UNITY_MODELS[model_name])
                if len(N_star) == 2:
                    # This model needs the line-with-end-markers handler
                    model_style = alpha_unity_style_dict[model_name]
                    handler_map[handle] = HandlerLineWithEndMarkers(
                        marker=model_style["marker"],
                        markersize=model_style["ms"] - 1,
                        color="k",
                        linewidth=model_style["lw"],
                        markerfacecolor=model_style["color"],
                        markeredgecolor=model_style["edgecolor"],
                    )
                break

        # Check for multiline scatter labels (containing newlines)
        if "\n" in label and handle not in handler_map:
            # Check if it's a scatter handle
            if hasattr(handle, "get_paths"):  # PathCollection from scatter
                # Get scatter properties if available
                try:
                    facecolors = handle.get_facecolors()
                    edgecolors = handle.get_edgecolors()
                    sizes = handle.get_sizes()

                    handler_map[handle] = HandlerScatterMultiline(
                        marker=handle.get_paths()[0] if handle.get_paths() else "o",
                        scatter_size=sizes[0] if len(sizes) > 0 else 50,
                        color=facecolors[0] if len(facecolors) > 0 else "k",
                        edgecolor=edgecolors[0] if len(edgecolors) > 0 else "k",
                        yoffset=5.25,
                    )
                except:
                    pass

    # Create legend with handler_map
    legend = ax.legend(
        handles=handles, labels=labels, handler_map=handler_map, **legend_kwargs
    )
    return legend
