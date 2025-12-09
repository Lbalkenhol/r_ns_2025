"""
Plot styling and configuration for r-ns plots.
Contains matplotlib settings, style dictionaries, and custom legend handlers.
"""

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import rc
from cycler import cycler
import seaborn as sns
import numpy as np


# ============================================================================
# Custom Legend Handlers
# ============================================================================


class HandlerTwoLines(HandlerBase):
    """Custom matplotlib legend handler for two-line entries."""

    def __init__(
        self, colors=["k", "k"], linestyles=["-", "--"], linewidths=[1.0, 1.0]
    ):
        self.colors = colors
        self.linestyles = linestyles
        self.linewidths = linewidths
        super().__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        """Create two horizontal lines, one above the other with separation."""
        lines = []

        # Top line (solid)
        y_top = ydescent + height * 0.65
        line1 = Line2D(
            [xdescent, xdescent + width],
            [y_top, y_top],
            color=self.colors[0],
            linestyle=self.linestyles[0],
            linewidth=self.linewidths[0],
            transform=trans,
        )

        # Bottom line (dashed) - more separated
        y_bottom = ydescent + height * 0.2
        line2 = Line2D(
            [xdescent, xdescent + width],
            [y_bottom, y_bottom],
            color=self.colors[1],
            linestyle=self.linestyles[1],
            linewidth=self.linewidths[1],
            transform=trans,
        )

        lines.extend([line1, line2])
        return lines


class HandlerLineWithEndMarkers(HandlerBase):
    """Custom matplotlib legend handler for line with markers at both ends."""

    def __init__(
        self,
        marker="s",
        markersize=6,
        color="k",
        linewidth=1.2,
        markerfacecolor=None,
        markeredgecolor=None,
    ):
        self.marker = marker
        self.markersize = markersize
        self.color = color
        self.linewidth = linewidth
        self.markerfacecolor = markerfacecolor if markerfacecolor is not None else color
        self.markeredgecolor = markeredgecolor if markeredgecolor is not None else color
        super().__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        """Create a line with markers at both ends."""
        # Create a line in the middle
        y_center = ydescent + height * 0.5
        line = Line2D(
            [xdescent, xdescent + width],
            [y_center, y_center],
            color=self.color,
            linestyle="-",
            linewidth=self.linewidth,
            transform=trans,
        )

        # Create markers at both ends
        x_left = xdescent + width * 0.1
        x_right = xdescent + width * 0.9

        marker_left = Line2D(
            [x_left],
            [y_center + height * 0.05],
            marker=self.marker,
            markersize=self.markersize,
            color=self.color,
            linestyle="",
            transform=trans,
            markerfacecolor=self.markerfacecolor,
            markeredgecolor=self.markeredgecolor,
        )

        marker_right = Line2D(
            [x_right],
            [y_center + height * 0.05],
            marker=self.marker,
            markersize=self.markersize,
            color=self.color,
            linestyle="",
            transform=trans,
            markerfacecolor=self.markerfacecolor,
            markeredgecolor=self.markeredgecolor,
        )

        return [line, marker_left, marker_right]


class HandlerMonomial(HandlerBase):
    """Custom legend handler for monomial potentials with e-fold shading."""

    def __init__(
        self,
        facecolor="0.8",
        edgecolor="0.6",
        linecolor="r",
        linewidth=1.2,
        yoffset=0.0,
    ):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.linecolor = linecolor
        self.linewidth = linewidth
        self.yoffset = yoffset
        super().__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Apply yoffset to all y-positions
        ydescent_adjusted = ydescent + self.yoffset

        # Create grey rectangle (filled)
        rect_fill = Rectangle(
            (xdescent, ydescent_adjusted),
            width,
            height,
            facecolor=self.facecolor,
            edgecolor="none",
            linewidth=0,
            transform=trans,
        )

        # Create diagonal red line from top-left to bottom-right
        x_start = xdescent + 0.025 * width
        y_start = ydescent_adjusted + 0.94 * height
        x_end = xdescent + 0.98 * width
        y_end = ydescent_adjusted + 0.037 * height

        line = Line2D(
            [x_start, x_end],
            [y_start, y_end],
            color=self.linecolor,
            linewidth=self.linewidth,
            transform=trans,
        )

        # Create rectangle edge on top
        rect_edge = Rectangle(
            (xdescent, ydescent_adjusted),
            width,
            height,
            facecolor="none",
            edgecolor=self.edgecolor,
            linewidth=0.6,
            transform=trans,
        )

        return [rect_fill, line, rect_edge]


class HandlerLineMultiline(HandlerBase):
    """Custom legend handler for line entries in multiline labels."""

    def __init__(self, linestyle=":", linewidth=1.0, color="k", yoffset=0.0):
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.color = color
        self.yoffset = yoffset
        super().__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create line at center with optional vertical offset
        y_center = ydescent + height * 0.5 + self.yoffset

        line = Line2D(
            [xdescent, xdescent + width],
            [y_center, y_center],
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            color=self.color,
            transform=trans,
        )

        return [line]


class HandlerScatterMultiline(HandlerBase):
    """Custom legend handler for scatter entries in multiline labels."""

    def __init__(
        self, marker="D", scatter_size=50, color="k", edgecolor="k", yoffset=0.0
    ):
        self.marker = marker
        # Convert scatter size (points²) to Line2D markersize (points)
        self.markersize = np.sqrt(scatter_size)
        self.color = color
        self.edgecolor = edgecolor
        self.yoffset = yoffset
        super().__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create marker at center with optional vertical offset
        x_center = xdescent + width * 0.5
        y_center = ydescent + height * 0.5 + self.yoffset

        marker = Line2D(
            [x_center],
            [y_center],
            marker=self.marker,
            markersize=self.markersize,
            markerfacecolor=self.color,
            markeredgecolor=self.edgecolor,
            linestyle="",
            transform=trans,
        )

        return [marker]


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
            "text.latex.preamble": r"\usepackage{amsmath, xfrac}",
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
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            # Color cycle
            "axes.prop_cycle": cycler(
                linestyle=["-", "--", "-.", ":", (0, (3, 5, 1, 5))] * 2,
                color=sns.color_palette("colorblind"),
            ),
            # Save settings
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.facecolor": "#FF000000",
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
        "colour": sns.color_palette("colorblind")[0],
        "ls": "-",
        "lw": 0.75,
        "filled": True,
        "label": "SPA+BK",
    },
    "SPA_BK_DESI": {
        "colour": sns.color_palette("colorblind")[-1],
        "ls": "-",
        "lw": 1.5,
        "filled": False,
        "label": "SPA+BK+DESI",
    },
    "FC": {"colour": "k", "ls": "-", "lw": 1.5, "filled": False, "label": "CMB 2030s"},
    "FC_DESI": {
        "colour": "k",
        "ls": "--",
        "lw": 1.5,
        "filled": False,
        "label": "CMB 2030s+DESI",
    },
}

# Style settings for specific inflation models
# N* values can be single values or [N_min, N_max] ranges
alpha_unity_style_dict = {
    "Starobinsky $R^2$": {
        "color": "k",
        "edgecolor": "k",
        "lw": 1.2,
        "ms": 6,
        "s": 60,
        "marker": "s",
    },
    "Higgs": {
        "color": "w",
        "edgecolor": "k",
        "lw": 1.2,
        "ms": 7,
        "s": 70,
        "marker": "o",
    },
}
