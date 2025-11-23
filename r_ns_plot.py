"""
Simple r-ns plot script.
Creates a clean plot with data constraints and basic theory elements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import getdist
from getdist import plots, loadMCSamples

from plot_style import style_dict
from theory_models import (
    add_concave_convex_divide,
    add_efold_shading,
)

# ============================================================================
# Load Data
# ============================================================================

chain_files = {
    "SPA_BK": "chains/SPA_BK/CLASS",
    "SPA_BK_DESI": "chains/SPA_BK_DESI/CLASS",
}

chains = {}
for key, value in chain_files.items():
    chains[key] = loadMCSamples(value, settings={"ignore_rows": 0.2})

# ============================================================================
# Create Plot
# ============================================================================

plt.close()

g = plots.get_single_plotter(width_inch=2.0 * 3.464, ratio=0.5)
g.settings.legend_frame = False
ax = plt.gca()

# Data constraints
all_dat = ["SPA_BK", "SPA_BK_DESI"]
g.plot_2d(
    [chains[dat] for dat in all_dat],
    ["ns", "r"],
    colors=[style_dict[dat]["colour"] for dat in all_dat],
    ls=[style_dict[dat]["ls"] for dat in all_dat],
    lws=[style_dict[dat]["lw"] for dat in all_dat],
    filled=[style_dict[dat]["filled"] for dat in all_dat],
)

# Add theory elements
add_efold_shading(ax, N_range=(50, 60), ns_range=(0.96, 1.0))
add_concave_convex_divide(ax, ns_range=(0.96, 1.0))

# Legend
g.add_legend(["SPA+BK", "SPA+BK+DESI"], fontsize=10)

# Axis setup
ax.set_ylim((0, 0.1))
ax.set_xlim((0.9575, 1.0))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylabel("Tensor-to-scalar ratio $r$")
ax.set_xlabel("Scalar spectral index $n_s$")

plt.tight_layout()

# plt.savefig("r_ns_plot.pdf", bbox_inches='tight', dpi=400)

plt.show()
