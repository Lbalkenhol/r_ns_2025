"""
Plot styling and configuration for r-ns plots.
Contains matplotlib settings, style dictionaries, and custom legend handlers.
"""

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib import rc
from cycler import cycler
import seaborn as sns


# ============================================================================
# Custom Legend Handler
# ============================================================================

class HandlerTwoLines(HandlerBase):
    """Custom matplotlib legend handler for two-line entries."""
    
    def __init__(self, colors=['k', 'k'], linestyles=['-', '--'], linewidths=[1., 1.]):
        self.colors = colors
        self.linestyles = linestyles
        self.linewidths = linewidths
        super().__init__()

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        """Create two horizontal lines, one above the other with separation."""
        lines = []
        
        # Top line (solid)
        y_top = ydescent + height * 0.65
        line1 = Line2D([xdescent, xdescent + width], [y_top, y_top],
                       color=self.colors[0], linestyle=self.linestyles[0], 
                       linewidth=self.linewidths[0], transform=trans)
        
        # Bottom line (dashed) - more separated
        y_bottom = ydescent + height * 0.2
        line2 = Line2D([xdescent, xdescent + width], [y_bottom, y_bottom],
                       color=self.colors[1], linestyle=self.linestyles[1],
                       linewidth=self.linewidths[1], transform=trans)
        
        lines.extend([line1, line2])
        return lines


# ============================================================================
# Plot Style Configuration
# ============================================================================

def set_plot_style(params=None, columnwidth=246, height_ratio=1.618):
    """
    Set the plotting style for publication-quality figures.
    
    Parameters
    ----------
    params : dict, optional
        Additional matplotlib parameters to override defaults
    columnwidth : float, optional
        Figure width in points (default: 246)
    height_ratio : float, optional
        Height to width ratio (default: 1.618, golden ratio)
    
    Returns
    -------
    dict
        Updated matplotlib rcParams
    """
    # Start with candl defaults (merged with our settings)
    rc("text", usetex=True)
    
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r'\usepackage{amsmath, xfrac}',
            
            # Figure size
            "figure.figsize": (columnwidth / 72, columnwidth / 72 / height_ratio),
            
            # Font sizes (candl values take precedence)
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            
            # Axes settings
            "axes.linewidth": 1.5,
            "axes.grid": False,
            "axes.xmargin": 0.02,
            
            # Legend
            "legend.frameon": False,
            
            # Tick settings (candl values where applicable)
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 1.,
            "ytick.major.width": 1.,
            "xtick.minor.width": 1.,
            "ytick.minor.width": 1.,
            
            # Color cycle
            "axes.prop_cycle": cycler(
                linestyle=["-", "--", "-.", ":", (0, (3, 5, 1, 5))]*2,
                color=sns.color_palette('colorblind')
            ),
            
            # Save settings
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.facecolor": '#FF000000',
        }
    )
    
    if params is not None:
        plt.rcParams.update(params)
    
    plt.clf()
    plt.close()
    
    return plt.rcParams


# Apply the plot style on import
set_plot_style()


# ============================================================================
# Style Dictionaries
# ============================================================================

# Style settings for different datasets
style_dict = {
    "SPA_BK": {
        "colour": sns.color_palette('colorblind')[0],
        "ls": "-",
        "lw": 0.75,
        "filled": True,
        "label": "SPA+BK"
    },
    "SPA_BK_DESI": {
        "colour": sns.color_palette('colorblind')[-1],
        "ls": "-",
        "lw": 1.5,
        "filled": False,
        "label": "SPA+BK+DESI"
    },
    "FC": {
        "colour": "k",
        "ls": "-",
        "lw": 1.5,
        "filled": False,
        "label": "CMB 2030s"
    },
    "FC_DESI": {
        "colour": "k",
        "ls": "--",
        "lw": 1.5,
        "filled": False,
        "label": "CMB 2030s+DESI"
    },
}

# Style settings for specific inflation models
alpha_unity_style_dict = {
    "Starobinsky $R^2$": {
        "color": "k",
        "s": 50,
        "marker": "s"
    },
    "Higgs": {
        "color": "w",
        "edgecolor": "k",
        "s": 70,
        "marker": "o"
    }
}