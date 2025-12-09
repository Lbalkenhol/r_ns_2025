"""
Streamlit app for interactive r-ns plots.
Run locally with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import getdist
from getdist import plots, loadMCSamples

from plot_style import style_dict
from theory_models import (
    add_concave_convex_divide,
    add_concave_convex_labels,
    add_efold_shading,
    add_alpha_unity_model_markers,
    add_polynomial_potentials,
)
from legend_utils import (
    create_dummy_plot_elements_for_legend,
    add_model_handlers_to_legend,
)

rgw_str = "r"
ns_str = "n_s"
N_star_str = "N_{\star}"

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(page_title="r-ns Plot Generator", page_icon="🎨", layout="wide")

st.title("Interactive r-ns Plot Generator")
st.markdown(
    "For best performance, wait for changes to appear in the plot before modifying further."
)
st.markdown(
    "When using this tool for publications (see export options at the bottom of the sidebar), please cite TBD, link to this webpage, and cite the appropriate publications for any data constraints you may be showing."
)

# ============================================================================
# Load Data (cached for performance)
# ============================================================================


@st.cache_data
def load_chains():
    """Load MCMC chains (cached to avoid reloading on every interaction)."""
    chain_files = {
        "SPA_BK": "chains/SPA_BK/CLASS",
        "SPA_BK_DESI": "chains/SPA_BK_DESI/CLASS",
    }

    chains = {}
    for key, value in chain_files.items():
        chains[key] = loadMCSamples(value, settings={"ignore_rows": 0.2})

    return chains


@st.cache_data
def create_forecast_chains():
    """Create forecast chains (cached)."""
    chains = load_chains()

    # Create forecast chains preserving r-ns correlation from real data
    cov = chains["SPA_BK"].cov(["r", "n_s"])
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    cov_scaled = np.diag([1e-3, 2e-3]) @ corr @ np.diag([1e-3, 2e-3])

    mrg = chains["SPA_BK"].getMargeStats()
    fc = getdist.gaussian_mixtures.GaussianND(
        [3e-3, mrg.parWithName("n_s").mean], cov_scaled, names=["r", "n_s"]
    )

    mrg_desi = chains["SPA_BK_DESI"].getMargeStats()
    fc_desi = getdist.gaussian_mixtures.GaussianND(
        [3e-3, mrg_desi.parWithName("n_s").mean], cov_scaled, names=["r", "n_s"]
    )

    return fc, fc_desi, corr


def create_custom_forecast(r_central, sigma_r, ns_central, sigma_ns, corr):
    """
    Create a custom forecast chain with user-specified parameters.

    Parameters
    ----------
    r_central : float
        Central value for r
    sigma_r : float
        Uncertainty on r
    ns_central : float
        Central value for ns
    sigma_ns : float
        Uncertainty on ns
    corr : array
        Correlation matrix from real data

    Returns
    -------
    GaussianND
        Custom forecast Gaussian distribution
    """
    cov_scaled = np.diag([sigma_r, sigma_ns]) @ corr @ np.diag([sigma_r, sigma_ns])
    return getdist.gaussian_mixtures.GaussianND(
        [r_central, ns_central], cov_scaled, names=["r", "n_s"]
    )


# Load data
try:
    chains = load_chains()
    fc, fc_desi, corr_matrix = create_forecast_chains()
    chains["FC"] = fc
    chains["FC_DESI"] = fc_desi
except Exception as e:
    st.error(f"Error loading chains: {e}")
    st.stop()

# ============================================================================
# Sidebar Controls
# ============================================================================

st.sidebar.header("Display")
plot_width = st.sidebar.slider("Plot width in browser window (%)", 50, 100, 80, step=5)

# Initialize session state for single_column if needed
if "single_column" not in st.session_state:
    st.session_state.single_column = False

single_column = st.sidebar.checkbox(
    "Single column style (half width)", value=st.session_state.single_column
)

# When single_column toggles, update the default aspect ratio
if single_column != st.session_state.single_column:
    st.session_state.single_column = single_column
    # Set the aspect ratio based on single_column state
    if single_column:
        st.session_state.aspect_ratio = 1.0
    else:
        st.session_state.aspect_ratio = 2.0
    st.rerun()

# Initialize aspect ratio if not exists
if "aspect_ratio" not in st.session_state:
    st.session_state.aspect_ratio = 2.0

aspect_ratio = st.sidebar.slider(
    "Aspect ratio", 1.0, 2.0, st.session_state.aspect_ratio, step=0.1
)

# Update session state with current aspect ratio (allows user override)
st.session_state.aspect_ratio = aspect_ratio

# Axis limits
st.sidebar.subheader("Axis Limits")

# Log scale option
use_log_scale = st.sidebar.checkbox("Log scale for $r$", value=False)

# Initialize session state if not exists
if "ns_min" not in st.session_state:
    st.session_state.ns_min = 0.9515
if "ns_max" not in st.session_state:
    st.session_state.ns_max = 1.0
if "r_min" not in st.session_state:
    st.session_state.r_min = 0.0
if "r_max" not in st.session_state:
    st.session_state.r_max = 0.1
if "use_log_scale" not in st.session_state:
    st.session_state.use_log_scale = False

# Update r_min based on log scale toggle
if use_log_scale != st.session_state.use_log_scale:
    if use_log_scale:
        st.session_state.r_min = 1e-3
        st.session_state.r_max = 0.1
    else:
        st.session_state.r_min = 0.0
        st.session_state.r_max = 0.1
    st.session_state.use_log_scale = use_log_scale

ns_min = st.sidebar.number_input(
    f"${ns_str}$ min",
    min_value=0.90,
    max_value=1.0,
    value=st.session_state.ns_min,
    format="%.4f",
    step=0.0025,
)
ns_max = st.sidebar.number_input(
    f"${ns_str}$ max",
    min_value=0.90,
    max_value=1.0,
    value=st.session_state.ns_max,
    format="%.4f",
    step=0.0025,
)

# Adjust r_min constraints based on log scale
r_min_constraint = 1e-4 if use_log_scale else 0.0
r_min = st.sidebar.number_input(
    f"${rgw_str}$ min",
    min_value=r_min_constraint,
    max_value=0.5,
    value=st.session_state.r_min,
    format="%.4e" if use_log_scale else "%.3f",
    step=1e-4 if use_log_scale else 0.002,
)
r_max = st.sidebar.number_input(
    f"${rgw_str}$ max",
    min_value=r_min_constraint,
    max_value=0.5,
    value=st.session_state.r_max,
    format="%.4e" if use_log_scale else "%.3f",
    step=1e-3 if use_log_scale else 0.002,
)

# Reset button
if st.sidebar.button("Reset to Default"):
    st.session_state.ns_min = 0.9515
    st.session_state.ns_max = 1.0
    if use_log_scale:
        st.session_state.r_min = 1e-3
        st.session_state.r_max = 0.1
    else:
        st.session_state.r_min = 0.0
        st.session_state.r_max = 0.1
    st.rerun()

# Update session state with current values (for next rerun)
st.session_state.ns_min = ns_min
st.session_state.ns_max = ns_max
st.session_state.r_min = r_min
st.session_state.r_max = r_max

# Validate ranges
if ns_min >= ns_max:
    st.sidebar.error("ns min must be less than ns max")
if r_min >= r_max:
    st.sidebar.error("r min must be less than r max")

# Dataset selection
st.sidebar.header("Constraints")
st.sidebar.subheader("Data")
show_spa_bk = st.sidebar.checkbox("SPA+BK", value=True)
show_spa_bk_desi = st.sidebar.checkbox("SPA+BK+DESI", value=True)

# Forecast selection
st.sidebar.subheader("Forecasts")
show_fc = st.sidebar.checkbox(
    f"CMB 2030s (${{\small {ns_str}=\mu^{{SPA+BK}}}}$)", value=False
)
st.sidebar.markdown(
    f"$${{\small {rgw_str}\sim\mathcal{{N}}(3\\times 10^{{-3}},10^{{-3}}),}}\\\\{{\small {ns_str}\sim\mathcal{{N}}(\mu^{{SPA+BK}}, 2\\times 10^{{-3}})}}$$"
)
show_fc_desi = st.sidebar.checkbox(
    f"CMB 2030s (${{\small {ns_str}=\mu^{{SPA+BK+DESI}}}}$)", value=False
)
st.sidebar.markdown(
    f"$${{\small {rgw_str}\sim\mathcal{{N}}(3\\times 10^{{-3}},10^{{-3}}),}}\\\\{{\small {ns_str}\sim\mathcal{{N}}(\mu^{{SPA+BK+DESI}}, 2\\times 10^{{-3}})}}$$"
)

# Custom forecast
show_custom = st.sidebar.checkbox("Custom Forecast", value=False)
if show_custom:
    st.sidebar.markdown("**Custom Forecast Parameters**")
    custom_r_central = st.sidebar.number_input(
        f"${rgw_str}$",
        min_value=0.0,
        max_value=0.5,
        value=3e-3,
        format="%.4e",
        step=1e-4,
    )
    custom_sigma_r = st.sidebar.number_input(
        f"$\sigma({rgw_str})$",
        min_value=1e-5,
        max_value=0.1,
        value=1e-3,
        format="%.4e",
        step=1e-4,
    )
    custom_ns_central = st.sidebar.number_input(
        f"${ns_str}$",
        min_value=0.90,
        max_value=1.10,
        value=0.965,
        format="%.4f",
        step=0.001,
    )
    custom_sigma_ns = st.sidebar.number_input(
        f"$\sigma({ns_str})$",
        min_value=1e-5,
        max_value=0.1,
        value=2e-3,
        format="%.4e",
        step=1e-4,
    )
    custom_label = st.sidebar.text_input("Legend label", value="Custom Forecast")

    # Create custom forecast chain
    chains["CUSTOM"] = create_custom_forecast(
        custom_r_central,
        custom_sigma_r,
        custom_ns_central,
        custom_sigma_ns,
        corr_matrix,
    )

    # Add to style_dict
    style_dict["CUSTOM"] = {
        "colour": "k",
        "ls": ":",
        "lw": 1.5,
        "filled": False,
        "label": custom_label,
    }

# Theory elements
st.sidebar.header("Theory Elements")
show_divide = st.sidebar.checkbox("Concave/Convex divide", value=False)
show_labels = st.sidebar.checkbox("Concave/Convex labels", value=False)
show_efold = st.sidebar.checkbox(
    f"Number of e-folds ${N_star_str}$ shading", value=True
)

# Always define N_min and N_max for use by polynomial potentials
if show_efold:
    N_min = st.sidebar.slider(f"${N_star_str}$ min", 40, 70, 50)
    N_max = st.sidebar.slider(f"${N_star_str}$ max", 40, 70, 60)
else:
    # Default values when e-fold shading is off
    N_min = 50
    N_max = 60

# Model markers
show_polynomial = st.sidebar.checkbox(f"Polynomial potentials", value=False)
st.sidebar.markdown(
    "$$\\small V(\phi) \propto \phi^{n},\, n=1, \\frac{2}{3}, \\frac{1}{3}$$"
)
show_starobinsky = st.sidebar.checkbox("Starobinsky $R^2$", value=False)
st.sidebar.markdown("$$\\small 42 \leq N_\star \leq 52$$")
show_higgs = st.sidebar.checkbox("Higgs inflation", value=False)

# Custom model marker
show_custom_marker = st.sidebar.checkbox("Custom Model Marker", value=False)
if show_custom_marker:
    st.sidebar.markdown("**Custom Model Parameters**")
    custom_marker_r = st.sidebar.number_input(
        f"${rgw_str}$",
        min_value=0.0,
        max_value=0.5,
        value=0.003,
        format="%.4e",
        step=1e-4,
        key="custom_marker_r",
    )
    custom_marker_ns = st.sidebar.number_input(
        f"${ns_str}$",
        min_value=0.90,
        max_value=1.10,
        value=0.965,
        format="%.4f",
        step=0.001,
        key="custom_marker_ns",
    )
    custom_marker_label = st.sidebar.text_input(
        "Model label", value="Custom Model", key="custom_marker_label"
    )

# Legend options
legend_fontsize = 10

# ============================================================================
# Generate Plot
# ============================================================================

# Determine which datasets to plot
all_dat = []
if show_spa_bk:
    all_dat.append("SPA_BK")
if show_spa_bk_desi:
    all_dat.append("SPA_BK_DESI")
if show_fc:
    all_dat.append("FC")
if show_fc_desi:
    all_dat.append("FC_DESI")
if show_custom:
    all_dat.append("CUSTOM")

# Create GetDist plotter (this properly initializes the figure)
plot_width_inch = 6.928 / 2 if single_column else 6.928
g = plots.get_single_plotter(width_inch=plot_width_inch, ratio=1 / aspect_ratio)
g.settings.legend_frame = False

# Plot data constraints using GetDist
if len(all_dat) > 0:
    g.plot_2d(
        [chains[dat] for dat in all_dat],
        ["n_s", "r"],
        colors=[style_dict[dat]["colour"] for dat in all_dat],
        ls=[style_dict[dat]["ls"] for dat in all_dat],
        lws=[style_dict[dat]["lw"] for dat in all_dat],
        filled=[style_dict[dat]["filled"] for dat in all_dat],
    )

# Get the axes from the GetDist plotter
ax = plt.gca()

# Add theory elements
if show_efold:
    add_efold_shading(ax, N_range=(N_min, N_max), ns_range=(ns_min, ns_max))

if show_divide:
    add_concave_convex_divide(ax, ns_range=(ns_min, ns_max))

if show_labels:
    add_concave_convex_labels(ax)

# Add polynomial potentials
if show_polynomial:
    add_polynomial_potentials(
        ax,
        p_values=[1 / 3, 2 / 3, 1],
        N_range=(N_min, N_max),
        add_labels=False,  # Don't add the on-panel text labels
        return_handles=False,
    )

# Add model markers - in desired legend order
models_to_show = []
if show_starobinsky:
    models_to_show.append("Starobinsky $R^2$")
if show_higgs:
    models_to_show.append("Higgs")

if len(models_to_show) > 0:
    add_alpha_unity_model_markers(ax, models=models_to_show, return_handles=False)

# Add custom model marker (will appear last)
if show_custom_marker:
    ax.scatter(
        custom_marker_ns,
        custom_marker_r,
        marker="*",
        s=200,
        c="orange",
        edgecolors="k",
        linewidths=0.5,
        zorder=10,
        label=custom_marker_label,
    )

# Add legend using the automatic handler function
# Add legend using the automatic handler function
if len(all_dat) > 0 or len(models_to_show) > 0 or show_polynomial or show_custom_marker:
    # Determine legend location based on log scale
    legend_loc = "lower right" if use_log_scale else "upper right"

    # Check if we have both FC and FC_DESI for special two-line legend entry
    if show_fc and show_fc_desi:
        # Use the advanced legend with two-line handler
        legend_handles, legend_labels, handler_map = (
            create_dummy_plot_elements_for_legend(all_dat, True)
        )

        # Get model handles from the plot and organize in desired order
        plot_handles, plot_labels = ax.get_legend_handles_labels()

        # Separate polynomial, Starobinsky, Higgs, and custom markers
        poly_handles = []
        poly_labels = []
        staro_handles = []
        staro_labels = []
        higgs_handles = []
        higgs_labels = []
        custom_handles = []
        custom_labels = []

        for h, l in zip(plot_handles, plot_labels):
            # Skip data constraint labels (already in legend_handles)
            if l in [style_dict[dat]["label"] for dat in all_dat if dat in style_dict]:
                continue

            # Categorize by label
            if l.startswith("$V(\\phi)"):  # Polynomial potentials
                poly_handles.append(h)
                poly_labels.append(l)
            elif l == "Starobinsky $R^2$":
                staro_handles.append(h)
                staro_labels.append(l)
            elif l == "Higgs":
                higgs_handles.append(h)
                higgs_labels.append(l)
            else:  # Custom marker
                custom_handles.append(h)
                custom_labels.append(l)

        # Add in desired order: data, polynomial, Starobinsky, Higgs, custom
        legend_handles.extend(poly_handles)
        legend_labels.extend(poly_labels)
        legend_handles.extend(staro_handles)
        legend_labels.extend(staro_labels)
        legend_handles.extend(higgs_handles)
        legend_labels.extend(higgs_labels)
        legend_handles.extend(custom_handles)
        legend_labels.extend(custom_labels)

        # Use the automatic handler function
        add_model_handlers_to_legend(
            ax,
            handles=legend_handles,
            labels=legend_labels,
            handler_map=handler_map,
            loc=legend_loc,
            fontsize=legend_fontsize,
            handlelength=1.5,
        )

    else:
        # Standard legend without two-line handler
        # Create dummy elements for data constraints (adds them to the plot)
        if len(all_dat) > 0:
            create_dummy_plot_elements_for_legend(all_dat, return_entries=False)

        # Now get ALL handles from the plot (including the dummy elements we just created)
        plot_handles, plot_labels = ax.get_legend_handles_labels()

        # Separate elements by type in desired order
        data_handles = []
        data_labels = []
        poly_handles = []
        poly_labels = []
        staro_handles = []
        staro_labels = []
        higgs_handles = []
        higgs_labels = []
        custom_handles = []
        custom_labels = []

        for h, l in zip(plot_handles, plot_labels):
            # Categorize each element
            if l in [style_dict[dat]["label"] for dat in all_dat if dat in style_dict]:
                # Data constraints
                data_handles.append(h)
                data_labels.append(l)
            elif l.startswith("$V(\\phi)"):  # Polynomial potentials
                poly_handles.append(h)
                poly_labels.append(l)
            elif l == "Starobinsky $R^2$":
                staro_handles.append(h)
                staro_labels.append(l)
            elif l == "Higgs":
                higgs_handles.append(h)
                higgs_labels.append(l)
            else:  # Custom marker
                custom_handles.append(h)
                custom_labels.append(l)

        # Combine in desired order: data, polynomial, Starobinsky, Higgs, custom
        all_handles = (
            data_handles + poly_handles + staro_handles + higgs_handles + custom_handles
        )
        all_labels = (
            data_labels + poly_labels + staro_labels + higgs_labels + custom_labels
        )

        # Use the automatic handler function (will detect Starobinsky and add handler)
        add_model_handlers_to_legend(
            ax,
            handles=all_handles,
            labels=all_labels,
            handler_map=None,  # No pre-existing handler_map
            loc=legend_loc,
            fontsize=legend_fontsize,
            handlelength=1.5,
        )

# Set axis properties
ax.set_ylim((r_min, r_max))
ax.set_xlim((ns_min, ns_max))

# Apply log scale if requested
if use_log_scale:
    ax.set_yscale("log")
    # Add minor ticks for log scale
    from matplotlib.ticker import LogLocator

    ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
else:
    ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylabel("Tensor-to-scalar ratio $r$")
ax.set_xlabel("Scalar spectral index $n_s$")

plt.tight_layout()

# Display plot in Streamlit with adjustable width
if plot_width == 100:
    # Full width - use single column
    st.pyplot(plt.gcf())
else:
    # Use three columns with proper ratios
    left_width = (100 - plot_width) / 2
    col1, col2, col3 = st.columns([left_width, plot_width, left_width])
    with col2:
        st.pyplot(plt.gcf())

# ============================================================================
# Download Options
# ============================================================================

st.sidebar.header("Export")

# Save buttons
if st.sidebar.button("Save as PDF"):
    plt.gcf().savefig("r_ns_plot.pdf", bbox_inches="tight", dpi=400)
    st.sidebar.success("Saved as r_ns_plot.pdf")

if st.sidebar.button("Save as PNG"):
    plt.gcf().savefig("r_ns_plot.png", bbox_inches="tight", dpi=400)
    st.sidebar.success("Saved as r_ns_plot.png")

# Export code button
if st.sidebar.button("Export Python Code"):
    # Generate the Python script based on current settings
    code = f'''"""
Generated r-ns plot script
Created by Interactive r-ns Plot Generator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

import getdist
from getdist import plots, loadMCSamples

from plot_style import style_dict
from theory_models import (
    add_concave_convex_divide, 
    add_concave_convex_labels,
    add_efold_shading,
    add_alpha_unity_model_markers,
    add_polynomial_potentials
)
from legend_utils import create_dummy_plot_elements_for_legend, add_model_handlers_to_legend

# ============================================================================
# Load Data
# ============================================================================

chain_files = {{
'''

    # Add chain loading based on selection
    if show_spa_bk or show_spa_bk_desi:
        if show_spa_bk:
            code += """    "SPA_BK": "chains/SPA_BK/CLASS",\n"""
        if show_spa_bk_desi:
            code += """    "SPA_BK_DESI": "chains/SPA_BK_DESI/CLASS",\n"""

    code += """}

chains = {}
for key, value in chain_files.items():
    chains[key] = loadMCSamples(value, settings={"ignore_rows": 0.2})

"""

    # Add forecast creation if needed
    if show_fc or show_fc_desi or show_custom:
        code += """# Create forecast chains preserving r-ns correlation from real data
cov = chains["SPA_BK"].cov(["r", "n_s"])
corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))

"""

        if show_fc:
            code += """cov_scaled = np.diag([1e-3, 2e-3]) @ corr @ np.diag([1e-3, 2e-3])
mrg = chains["SPA_BK"].getMargeStats()
chains["FC"] = getdist.gaussian_mixtures.GaussianND([3e-3, mrg.parWithName("n_s").mean],
                                                    cov_scaled,
                                                    names=["r", "n_s"])

"""

        if show_fc_desi:
            code += """cov_scaled = np.diag([1e-3, 2e-3]) @ corr @ np.diag([1e-3, 2e-3])
mrg_desi = chains["SPA_BK_DESI"].getMargeStats()
chains["FC_DESI"] = getdist.gaussian_mixtures.GaussianND([3e-3, mrg_desi.parWithName("n_s").mean],
                                                         cov_scaled,
                                                         names=["r", "n_s"])

"""

        if show_custom:
            code += f"""# Custom forecast
cov_scaled = np.diag([{custom_sigma_r}, {custom_sigma_ns}]) @ corr @ np.diag([{custom_sigma_r}, {custom_sigma_ns}])
chains["CUSTOM"] = getdist.gaussian_mixtures.GaussianND([{custom_r_central}, {custom_ns_central}],
                                                        cov_scaled,
                                                        names=["r", "n_s"])

style_dict["CUSTOM"] = {{
    "colour": "k",
    "ls": ":",
    "lw": 1.5,
    "filled": False,
    "label": "{custom_label}"
}}

"""

    code += """# ============================================================================
# Create Plot
# ============================================================================

plt.close()

# Create GetDist plotter
g = plots.get_single_plotter(width_inch="""

    code += f"""{plot_width_inch:.3f}, ratio={1/aspect_ratio:.2f})
g.settings.legend_frame = False

# Plot data constraints
all_dat = ["""

    # Add dataset list
    dat_list = []
    if show_spa_bk:
        dat_list.append('"SPA_BK"')
    if show_spa_bk_desi:
        dat_list.append('"SPA_BK_DESI"')
    if show_fc:
        dat_list.append('"FC"')
    if show_fc_desi:
        dat_list.append('"FC_DESI"')
    if show_custom:
        dat_list.append('"CUSTOM"')

    code += (
        ", ".join(dat_list)
        + """]

g.plot_2d([chains[dat] for dat in all_dat],
          ["n_s", "r"],
          colors=[style_dict[dat]["colour"] for dat in all_dat],
          ls=[style_dict[dat]["ls"] for dat in all_dat],
          lws=[style_dict[dat]["lw"] for dat in all_dat],
          filled=[style_dict[dat]["filled"] for dat in all_dat])

ax = plt.gca()

"""
    )

    # Add theory elements
    if show_efold:
        code += f"""# Add e-fold shading
add_efold_shading(ax, N_range=({N_min}, {N_max}), ns_range=({ns_min}, {ns_max}))

"""

    if show_divide:
        code += f"""# Add concave/convex divide
add_concave_convex_divide(ax, ns_range=({ns_min}, {ns_max}))

"""

    if show_labels:
        code += """# Add concave/convex labels
add_concave_convex_labels(ax)

"""

    if show_polynomial:
        code += f"""# Add polynomial potentials
add_polynomial_potentials(ax, p_values=[1/3, 2/3, 1], N_range=({N_min}, {N_max}),
                         add_labels=False, return_handles=False)

"""

    # Add model markers
    if show_starobinsky or show_higgs:
        code += """# Add model markers
models_to_show = ["""
        model_list = []
        if show_starobinsky:
            model_list.append('"Starobinsky $R^2$"')
        if show_higgs:
            model_list.append('"Higgs"')
        code += (
            ", ".join(model_list)
            + """]
add_alpha_unity_model_markers(ax, models=models_to_show, return_handles=False)

"""
        )

    if show_custom_marker:
        code += f"""# Add custom model marker
ax.scatter({custom_marker_ns}, {custom_marker_r}, 
          marker='*', s=200, c='orange', edgecolors='k', linewidths=0.5,
          zorder=10, label="{custom_marker_label}")

"""

    # Add legend
    legend_loc = "lower right" if use_log_scale else "upper right"

    if show_fc and show_fc_desi:
        code += f"""# Create legend with two-line handler
legend_handles, legend_labels, handler_map = create_dummy_plot_elements_for_legend(all_dat, True)
plot_handles, plot_labels = ax.get_legend_handles_labels()
for h, l in zip(plot_handles, plot_labels):
    if l not in [style_dict[dat]["label"] for dat in all_dat if dat in style_dict]:
        legend_handles.append(h)
        legend_labels.append(l)

add_model_handlers_to_legend(ax, handles=legend_handles, labels=legend_labels,
                             handler_map=handler_map, loc="{legend_loc}",
                             fontsize={legend_fontsize}, handlelength=1.5)

"""
    else:
        code += f"""# Create legend
create_dummy_plot_elements_for_legend(all_dat)
add_model_handlers_to_legend(ax, loc="{legend_loc}", fontsize={legend_fontsize}, handlelength=1.5)

"""

    # Axis setup
    code += f"""# Axis setup
ax.set_ylim(({r_min}, {r_max}))
ax.set_xlim(({ns_min}, {ns_max}))

"""

    if use_log_scale:
        code += """# Log scale
ax.set_yscale("log")
ax.yaxis.set_minor_locator(LogLocator(subs='auto'))

"""
    else:
        code += """ax.yaxis.set_minor_locator(AutoMinorLocator())

"""

    code += """ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylabel("Tensor-to-scalar ratio $r$")
ax.set_xlabel("Scalar spectral index $n_s$")

plt.tight_layout()
plt.savefig("r_ns_plot.pdf", bbox_inches='tight', dpi=400)
plt.show()
"""

    # Save the code to file
    with open("custom_rns_plot_script.py", "w") as f:
        f.write(code)

    st.sidebar.success("Saved as custom_rns_plot_script.py")
