"""
Utilities for handling legends with GetDist integration.
"""

import matplotlib.pyplot as plt
from plot_style import style_dict, HandlerTwoLines


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
                dummy_handle_combined = plt.Line2D([], [], color='k')
                legend_handles.append(dummy_handle_combined)
                legend_labels.append("CMB 2030s")
                handler_map[dummy_handle_combined] = HandlerTwoLines()
                continue
            elif dat == "FC_DESI":
                # Skip - already handled with FC
                continue
        
        # Regular entries
        if style_dict[dat]["filled"]:
            handle = ax.fill_between([2, 2], [2, 2], [2, 2],
                                    color=style_dict[dat]["colour"],
                                    lw=0,
                                    label=style_dict[dat]["label"])
        else:
            handle = ax.plot([2, 2], [2, 2],
                           color=style_dict[dat]["colour"],
                           ls=style_dict[dat]["ls"],
                           label=style_dict[dat]["label"])[0]
        
        legend_handles.append(handle)
        legend_labels.append(style_dict[dat]["label"])
    
    if return_entries:
        return legend_handles, legend_labels, handler_map
    return None
