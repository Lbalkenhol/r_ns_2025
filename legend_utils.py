"""
Utilities for handling legends with GetDist integration.
"""

import matplotlib.pyplot as plt
from plot_style import (
    style_dict,
    HandlerTwoLines,
    HandlerLineWithEndMarkers,
    alpha_unity_style_dict,
)
from theory_models import ALPHA_UNITY_MODELS
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


def add_model_handlers_to_legend(
    ax, handles=None, labels=None, handler_map=None, **legend_kwargs
):
    """
    Add legend with automatic handling of special model entries (Starobinsky, etc.).

    This function gets all handles and labels from the current axes (or uses provided ones),
    identifies models that need special handlers (like Starobinsky R²), and creates a legend
    with the appropriate custom handlers.

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
        # Check if this is a model with N* range (use ALPHA_UNITY_MODELS as source of truth)
        if label in ALPHA_UNITY_MODELS and label in alpha_unity_style_dict:
            N_star = np.atleast_1d(ALPHA_UNITY_MODELS[label])
            if len(N_star) > 1:
                # This model needs the line-with-end-markers handler
                model_style = alpha_unity_style_dict[label]
                handler_map[handle] = HandlerLineWithEndMarkers(
                    marker=model_style["marker"],
                    markersize=model_style["ms"] - 1,
                    color=model_style["color"],
                    linewidth=model_style["lw"],
                )

    # Create legend with handler_map
    legend = ax.legend(
        handles=handles, labels=labels, handler_map=handler_map, **legend_kwargs
    )
    return legend
