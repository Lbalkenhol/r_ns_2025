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
    add_efold_shading_monomial,
    add_alpha_unity_model_markers,
    add_monomial_potentials,
    add_polynomial_alpha_attractor,
    ALPHA_UNITY_MODELS,
)
from legend_utils import (
    create_dummy_plot_elements_for_legend,
    add_model_handlers_to_legend,
    add_monomial_legend_entry,
    add_alpha_attractor_legend_entry,
)

rgw_str = "r"
ns_str = "n_s"
N_star_str = "N_{\\star}"

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(page_title="r-ns Plot Generator", page_icon="🎨", layout="wide")

st.title("Interactive r-ns Plot Generator")
st.markdown(
    "For best performance, wait for changes to appear in the plot before modifying further. The resolution is kept low to boost performance; downloads (pdf, png) are high res."
)
st.markdown(
    "When using this tool for publications (see export options at the bottom of the sidebar), please cite [Balkenhol et al. 2025](https://arxiv.org/abs/2512.10613), link to this webpage, and cite the appropriate publications for any data constraints you may be showing."
)

# Set reasonable limits for image rendering to prevent crashes
import matplotlib
from PIL import Image
import shutil
import os

# Increase PIL's pixel limit to prevent DecompressionBomb errors
Image.MAX_IMAGE_PIXELS = 80_000_000

matplotlib.rcParams["figure.max_open_warning"] = 0
# Limit DPI for display (exports can use higher DPI)
DISPLAY_DPI = 150  # Lower DPI for Streamlit display


def clear_tex_cache():
    """Clear matplotlib's TeX cache to fix corrupted font files."""
    try:
        tex_cache = matplotlib.get_cachedir()
        tex_dir = os.path.join(tex_cache, "tex.cache")
        if os.path.exists(tex_dir):
            shutil.rmtree(tex_dir)
        # Also clear any .dvi files in the cache
        for f in os.listdir(tex_cache):
            if f.endswith((".dvi", ".tex", ".log", ".aux")):
                os.remove(os.path.join(tex_cache, f))
    except Exception:
        pass  # Silently ignore if we can't clear cache


# Track if \sfrac rendering has failed (use fallback fractions)
if "sfrac_failed" not in st.session_state:
    st.session_state.sfrac_failed = False


def get_monomial_label(N_min, N_max):
    """Get monomial legend label, with fallback if sfrac fails. sfrac is seeming to cause some trouble so it has been taken out for now."""
    if st.session_state.sfrac_failed:
        # Fallback: use regular fractions
        return (
            r"$V(\phi) \propto \phi^{n},\, n=1, 2/3, 1/3$"
            + f"\n(${N_min}\\leq\\! N_\\star\\!\\leq {N_max}$)"
        )
    else:
        # Preferred: use sfrac for nicer inline fractions
        return (
            r"$V(\phi) \propto \phi^{n},\, n=1, 2/3, 1/3$"
            + f"\n(${N_min}\\leq\\! N_\\star\\!\\leq {N_max}$)"
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
    st.session_state.ns_min = 0.954
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
        st.session_state.r_min = 3e-4
        st.session_state.r_max = 1e-1
    else:
        st.session_state.r_min = 0.0
        st.session_state.r_max = 0.1
    st.session_state.use_log_scale = use_log_scale
    st.rerun()

# Additional safety check: if log scale is on but r_min is invalid, fix it
if use_log_scale and st.session_state.r_min <= 0:
    st.session_state.r_min = 3e-4
    st.rerun()

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
    st.session_state.ns_min = 0.954
    st.session_state.ns_max = 1.0
    if use_log_scale:
        st.session_state.r_min = 3e-4
        st.session_state.r_max = 1e-1
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
    f"CMB 2030s (${{\\small {ns_str}=\\mu^{{SPA+BK}}}}$)", value=False
)
st.sidebar.markdown(
    f"$${{\\small {rgw_str}\\sim\\mathcal{{N}}(3\\times 10^{{-3}},10^{{-3}}),}}\\\\{{\\small {ns_str}\\sim\\mathcal{{N}}(\\mu^{{SPA+BK}}, 2\\times 10^{{-3}})}}$$"
)
show_fc_desi = st.sidebar.checkbox(
    f"CMB 2030s (${{\\small {ns_str}=\\mu^{{SPA+BK+DESI}}}}$)", value=False
)
st.sidebar.markdown(
    f"$${{\\small {rgw_str}\\sim\\mathcal{{N}}(3\\times 10^{{-3}},10^{{-3}}),}}\\\\{{\\small {ns_str}\\sim\\mathcal{{N}}(\\mu^{{SPA+BK+DESI}}, 2\\times 10^{{-3}})}}$$"
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
        f"$\\sigma({rgw_str})$",
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
        f"$\\sigma({ns_str})$",
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

# ============================================================================
# Theory Elements
# ============================================================================

st.sidebar.header("Theory Elements")

# --- Concave/Convex ---
st.sidebar.subheader("Concave/Convex")
show_divide = st.sidebar.checkbox("Dividing line", value=True)
if show_divide:
    show_labels = st.sidebar.checkbox("Labels", value=False)
else:
    show_labels = False

# --- Monomial Potentials ---
st.sidebar.subheader("Monomial Potentials")
show_monomial = st.sidebar.checkbox("Show monomial potentials", value=True)
if show_monomial:
    st.sidebar.markdown("$$\\small V(\\phi) \\propto \\phi^{n}$$")
    show_efold = st.sidebar.checkbox(f"$${N_star_str}$$ shading", value=True)
    show_monomial_lines = st.sidebar.checkbox("Example potentials", value=False)
    st.sidebar.markdown("$$\\small n=1, \\frac{2}{3}, \\frac{1}{3}$$")

    if show_efold or show_monomial_lines:
        N_min = st.sidebar.slider(
            f"${N_star_str}$ min", 40, 65, 47, key="monomial_N_min"
        )
        N_max = st.sidebar.slider(
            f"${N_star_str}$ max", 40, 65, 57, key="monomial_N_max"
        )
    else:
        N_min = 47
        N_max = 57
else:
    show_efold = False
    show_monomial_lines = False
    N_min = 47
    N_max = 57

# --- Starobinsky R² ---
st.sidebar.subheader("Starobinsky $R^2$")
show_starobinsky = st.sidebar.checkbox("Show Starobinsky $R^2$", value=False)
if show_starobinsky:
    starobinsky_mode = st.sidebar.radio(
        "",  # Empty label - removed "Display mode"
        ["Single $N_\\star$", "$N_\\star$ range"],
        key="starobinsky_mode",
        label_visibility="collapsed",
        horizontal=True,
    )
    if starobinsky_mode == "Single $N_\\star$":
        starobinsky_N = st.sidebar.slider(
            f"${N_star_str}$", 40, 60, 51, key="starobinsky_single"
        )
        ALPHA_UNITY_MODELS["Starobinsky $R^2$"] = starobinsky_N
    else:
        starobinsky_N_min = st.sidebar.slider(
            f"${N_star_str}$ min", 40, 60, 47, key="starobinsky_min"
        )
        starobinsky_N_max = st.sidebar.slider(
            f"${N_star_str}$ max", 40, 60, 57, key="starobinsky_max"
        )
        ALPHA_UNITY_MODELS["Starobinsky $R^2$"] = [starobinsky_N_min, starobinsky_N_max]

# --- Higgs Inflation ---
st.sidebar.subheader("Higgs Inflation")
show_higgs = st.sidebar.checkbox("Show Higgs inflation", value=False)
if show_higgs:
    higgs_mode = st.sidebar.radio(
        "",  # Empty label - removed "Display mode"
        ["Single $N_\\star$", "$N_\\star$ range"],
        label_visibility="collapsed",
        key="higgs_mode",
        horizontal=True,
    )
    if higgs_mode == "Single $N_\\star$":
        higgs_N = st.sidebar.slider(f"${N_star_str}$", 40, 60, 55, key="higgs_single")
        ALPHA_UNITY_MODELS["Higgs"] = higgs_N
    else:
        higgs_N_min = st.sidebar.slider(
            f"${N_star_str}$ min", 40, 60, 47, key="higgs_min"
        )
        higgs_N_max = st.sidebar.slider(
            f"${N_star_str}$ max", 40, 60, 57, key="higgs_max"
        )
        ALPHA_UNITY_MODELS["Higgs"] = [higgs_N_min, higgs_N_max]

# --- Polynomial α-attractor ---
st.sidebar.subheader("Polynomial α-attractor")
show_alpha_attractor = st.sidebar.checkbox("Show polynomial α-attractor", value=False)
if show_alpha_attractor:
    st.sidebar.markdown("$$\\small V(\\phi) \\propto |\\phi|^k/(\\mu^k + |\\phi|^k)$$")
    alpha_k = st.sidebar.selectbox("Power $k$", [1, 2, 3, 4], index=1)
    alpha_num_lines = st.sidebar.radio(
        "Number of lines", [1, 2], index=1, horizontal=True
    )

    alpha_N_values = []
    if alpha_num_lines >= 1:
        alpha_N_1 = st.sidebar.slider(
            f"${N_star_str}$ (line 1)", 40, 65, 47, key="alpha_N_1"
        )
        alpha_N_values.append(alpha_N_1)
    if alpha_num_lines >= 2:
        alpha_N_2 = st.sidebar.slider(
            f"${N_star_str}$ (line 2)", 40, 65, 57, key="alpha_N_2"
        )
        alpha_N_values.append(alpha_N_2)

# --- Custom Model Marker ---
st.sidebar.subheader("Custom Model Marker")
show_custom_marker = st.sidebar.checkbox("Show custom model marker", value=False)
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
        "Model label",
        value="Custom Model $V(\\varphi)\\propto\\dots$",
        key="custom_marker_label",
    )

# Legend options
legend_fontsize = 9

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

# Close any existing figures to prevent memory accumulation
plt.close("all")

# Create GetDist plotter (this properly initializes the figure)
plot_width_inch = 6.928 / 2 if single_column else 6.928
g = plots.get_single_plotter(width_inch=plot_width_inch, ratio=1 / aspect_ratio)
g.settings.legend_frame = False
g.settings.tight_layout = False

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

# Add theory elements in correct order for proper layering
# Layering (back to front): grey shading -> divide line -> red lines -> data -> ticks

# Grey shading (lowest background)
if show_efold:
    add_efold_shading_monomial(
        ax, N_range=(N_min, N_max), ns_range=(ns_min, ns_max), zorder=-10
    )

# Concave/convex divide (above shading)
if show_divide:
    add_concave_convex_divide(ax, ns_range=(ns_min, ns_max), zorder=-5)

# Concave/convex labels
if show_labels:
    add_concave_convex_labels(ax)

# Red monomial lines (above divide, but still behind data)
if show_monomial_lines:
    add_monomial_potentials(
        ax,
        p_values=[1 / 3, 2 / 3, 1],
        N_range=(N_min, N_max),
        add_labels=False,
        return_handles=False,
        zorder=-1,
    )

# Add model markers - in desired legend order
models_to_show = []
if show_starobinsky:
    models_to_show.append("Starobinsky $R^2$")
if show_higgs:
    models_to_show.append("Higgs")

if len(models_to_show) > 0:
    add_alpha_unity_model_markers(ax, models=models_to_show, return_handles=False)

# Add polynomial alpha-attractor lines
if show_alpha_attractor and len(alpha_N_values) > 0:
    add_polynomial_alpha_attractor(
        ax, N_star=alpha_N_values, k=alpha_k, return_handles=False
    )

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

# ============================================================================
# Legend Construction
# ============================================================================

# Determine if we need the advanced legend system
needs_advanced_legend = (
    len(all_dat) > 0
    or show_monomial_lines
    or show_efold
    or len(models_to_show) > 0
    or (show_alpha_attractor and len(alpha_N_values) > 0)
    or show_custom_marker
)

if needs_advanced_legend:
    # Determine legend location based on log scale
    legend_loc = "lower right" if use_log_scale else "upper right"

    # Check if we have both FC and FC_DESI for special two-line legend entry
    if show_fc and show_fc_desi:
        # Use the advanced legend with two-line handler
        legend_handles, legend_labels, handler_map = (
            create_dummy_plot_elements_for_legend(all_dat, True)
        )
    elif len(all_dat) > 0:
        # Regular data constraints
        legend_handles, legend_labels, handler_map = (
            create_dummy_plot_elements_for_legend(all_dat, True)
        )
    else:
        # No data constraints
        legend_handles = []
        legend_labels = []
        handler_map = {}

    # Add monomial potential entry if requested
    if show_monomial_lines or show_efold:
        # Determine y-offset based on whether we have multiline entries below
        has_multiline_below = (
            (show_starobinsky and starobinsky_mode == "Single $N_\\star$")
            or (show_higgs and higgs_mode == "Single $N_\\star$")
            or (show_alpha_attractor and len(alpha_N_values) > 0)
        )
        yoffset = 5.25 if has_multiline_below else 0.0

        # Choose handler based on what's shown
        if show_monomial_lines and show_efold:
            # Both: use custom handler with red line and grey shading
            add_monomial_legend_entry(
                legend_handles,
                legend_labels,
                handler_map,
                N_range=(N_min, N_max),
                yoffset=yoffset,
                use_sfrac=False,
            )
        elif show_efold:
            # Only shading: show grey rectangle with V∝φⁿ (no specific exponents)
            from matplotlib.patches import Rectangle

            dummy_handle = Rectangle((0, 0), 1, 1, facecolor="0.8", edgecolor="0.6")
            legend_handles.append(dummy_handle)
            legend_labels.append(
                f"$V(\\phi) \\propto \\phi^n$\n(${N_min}\\leq\\! N_\\star\\!\\leq {N_max}$)"
            )
        else:
            # Only monomial lines: show red line with specific exponents
            from matplotlib.lines import Line2D

            dummy_handle = Line2D([], [], color="r", lw=1.2)
            legend_handles.append(dummy_handle)
            legend_labels.append(get_monomial_label(N_min, N_max))

    # Get model handles from the plot
    model_handles, model_labels = add_alpha_unity_model_markers(
        ax, models=models_to_show, return_handles=True
    )
    if model_handles:
        legend_handles.extend(model_handles)
        legend_labels.extend(model_labels)

    # Add polynomial alpha-attractor entry if requested
    if show_alpha_attractor and len(alpha_N_values) > 0:
        # Determine y-offset (0 if >2 N* values, otherwise 5.25)
        yoffset = 0.0 if len(alpha_N_values) > 2 else 5.25
        add_alpha_attractor_legend_entry(
            legend_handles,
            legend_labels,
            handler_map,
            N_star=alpha_N_values,
            k=alpha_k,
            yoffset=yoffset,
        )

    # Add custom marker handle if present
    if show_custom_marker:
        # Get the custom marker from the plot
        plot_handles, plot_labels = ax.get_legend_handles_labels()
        for h, l in zip(plot_handles, plot_labels):
            if l == custom_marker_label:
                legend_handles.append(h)
                legend_labels.append(l)
                break

    # Create the legend with all handlers
    add_model_handlers_to_legend(
        ax,
        handles=legend_handles,
        labels=legend_labels,
        handler_map=handler_map,
        loc=legend_loc,
        fontsize=legend_fontsize,
        handlelength=1.5,
        ncol=1,
    )

# Apply log scale if requested (MUST be done BEFORE setting limits)
if use_log_scale:
    ax.set_yscale("log")
    # Validate that r_min is positive for log scale
    if r_min <= 0:
        r_min = 3e-4  # Safe default for log scale
    # Add minor ticks for log scale
    from matplotlib.ticker import LogLocator

    ax.yaxis.set_minor_locator(LogLocator(subs="auto"))
else:
    ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.xaxis.set_minor_locator(AutoMinorLocator())

# Set axis properties (AFTER setting scale)
ax.set_ylim((r_min, r_max))
ax.set_xlim((ns_min, ns_max))

# Set axis ticks and spines to foreground
ax.set_axisbelow(False)
for spine in ax.spines.values():
    spine.set_zorder(1000)
ax.xaxis.set_zorder(1000)
ax.yaxis.set_zorder(1000)

ax.set_ylabel("Tensor-to-scalar ratio $r$")
ax.set_xlabel("Scalar spectral index $n_s$")

# Try tight_layout, but don't fail if it doesn't work with GetDist
try:
    plt.tight_layout()
except:
    pass

# Check figure size and prevent rendering if too large
fig = plt.gcf()
fig_width, fig_height = fig.get_size_inches()
estimated_pixels = (fig_width * DISPLAY_DPI) * (fig_height * DISPLAY_DPI)

# Limit to ~50 million pixels (well under PIL's 178M limit)
MAX_SAFE_PIXELS = 50_000_000

if estimated_pixels > MAX_SAFE_PIXELS:
    st.error(
        f"Plot is too large to display ({estimated_pixels/1e6:.1f}M pixels). "
        f"Please reduce the plot size or aspect ratio. "
        f"Maximum safe size: {MAX_SAFE_PIXELS/1e6:.1f}M pixels."
    )
    st.stop()


def display_plot(container=None):
    """Display the plot with error handling for TeX cache corruption."""
    kwargs = dict(
        clear_figure=True,
        dpi=DISPLAY_DPI,
        width="stretch",
        facecolor="white",
    )
    target = container if container else st

    try:
        target.pyplot(plt.gcf(), **kwargs)
    except ValueError as e:
        if "vf file" in str(e) or "Misplaced packet" in str(e):
            # TeX cache corruption - clear cache and fall back to simpler fractions
            clear_tex_cache()
            if not st.session_state.sfrac_failed:
                # First failure: switch to fallback fractions and retry
                st.session_state.sfrac_failed = True
                st.warning(
                    "Detected font rendering issue, switching to fallback mode..."
                )
                st.rerun()
            else:
                # Already in fallback mode - show error
                st.error("LaTeX rendering error. Please try refreshing the page.")
                raise
        else:
            raise


# ============================================================================
# Prepare Export Buffers (BEFORE displaying, since display clears the figure)
# ============================================================================

import io

# Get the current figure dimensions for export
fig = plt.gcf()
fig_width, fig_height = fig.get_size_inches()

# Set reasonable DPI limits to avoid oversized images
max_dpi = min(400, int(65000 / max(fig_width, fig_height)))

# Save figure to buffers BEFORE displaying (display_plot clears the figure)
pdf_buffer = io.BytesIO()
fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", dpi=max_dpi)
pdf_buffer.seek(0)

png_buffer = io.BytesIO()
fig.savefig(png_buffer, format="png", bbox_inches="tight", dpi=max_dpi)
png_buffer.seek(0)

# Display plot in Streamlit with adjustable width and controlled DPI
# Use lower DPI for display to prevent memory issues on Streamlit Cloud
if plot_width == 100:
    # Full width - use single column
    display_plot()
else:
    # Use three columns with proper ratios
    left_width = (100 - plot_width) / 2
    col1, col2, col3 = st.columns([left_width, plot_width, left_width])
    with col2:
        display_plot(col2)

# ============================================================================
# Download Options
# ============================================================================

st.sidebar.header("Export")

# Generate the Python script code (we'll need this for the download button)
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
    add_efold_shading_monomial,
    add_alpha_unity_model_markers,
    add_monomial_potentials,
    add_polynomial_alpha_attractor,
    ALPHA_UNITY_MODELS,
)
from legend_utils import (
    create_dummy_plot_elements_for_legend, 
    add_model_handlers_to_legend,
    add_monomial_legend_entry,
    add_alpha_attractor_legend_entry,
)

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

# Update ALPHA_UNITY_MODELS if needed
if show_starobinsky:
    if starobinsky_mode == "Single $N_\\star$":
        code += f"""# Update Starobinsky N*
ALPHA_UNITY_MODELS["Starobinsky $R^2$"] = {starobinsky_N}

"""
    else:
        code += f"""# Update Starobinsky N* range
ALPHA_UNITY_MODELS["Starobinsky $R^2$"] = [{starobinsky_N_min}, {starobinsky_N_max}]

"""

if show_higgs:
    if higgs_mode == "Single $N_\\star$":
        code += f"""# Update Higgs N*
ALPHA_UNITY_MODELS["Higgs"] = {higgs_N}

"""
    else:
        code += f"""# Update Higgs N* range
ALPHA_UNITY_MODELS["Higgs"] = [{higgs_N_min}, {higgs_N_max}]

"""

code += """# ============================================================================
# Create Plot
# ============================================================================

plt.close()

# Create GetDist plotter
g = plots.get_single_plotter(width_inch="""

code += f"""{plot_width_inch:.3f}, ratio={1/aspect_ratio:.2f})
g.settings.legend_frame = False
g.settings.tight_layout = False

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

if len(all_dat) > 0:
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
    code += f"""# Add e-fold shading for monomial potentials (lowest background)
add_efold_shading_monomial(ax, N_range=({N_min}, {N_max}), ns_range=({ns_min}, {ns_max}), zorder=-10)

"""

if show_divide:
    code += f"""# Add concave/convex divide (above shading)
add_concave_convex_divide(ax, ns_range=({ns_min}, {ns_max}), zorder=-5)

"""

if show_labels:
    code += """# Add concave/convex labels
add_concave_convex_labels(ax)

"""

if show_monomial_lines:
    code += f"""# Add monomial potentials (above divide, behind data)
add_monomial_potentials(ax, p_values=[1/3, 2/3, 1], N_range=({N_min}, {N_max}),
                         add_labels=False, return_handles=False, zorder=-1)

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

# Add polynomial alpha-attractor
if show_alpha_attractor and len(alpha_N_values) > 0:
    code += f"""# Add polynomial alpha-attractor lines
add_polynomial_alpha_attractor(ax, N_star={alpha_N_values}, k={alpha_k})

"""

if show_custom_marker:
    code += f"""# Add custom model marker
ax.scatter({custom_marker_ns}, {custom_marker_r}, 
          marker='*', s=200, c='orange', edgecolors='k', linewidths=0.5,
          zorder=10, label=r"{custom_marker_label}")

"""

# Add legend construction
legend_loc = "lower right" if use_log_scale else "upper right"

if needs_advanced_legend:
    code += f"""# Build legend
"""
    if show_fc and show_fc_desi:
        code += """legend_handles, legend_labels, handler_map = create_dummy_plot_elements_for_legend(all_dat, True)
"""
    elif len(all_dat) > 0:
        code += """legend_handles, legend_labels, handler_map = create_dummy_plot_elements_for_legend(all_dat, True)
"""
    else:
        code += """legend_handles, legend_labels, handler_map = [], [], {}
"""

    if show_monomial_lines or show_efold:
        has_multiline_below = (
            (show_starobinsky and starobinsky_mode == "Single $N_\\star$")
            or (show_higgs and higgs_mode == "Single $N_\\star$")
            or (show_alpha_attractor and len(alpha_N_values) > 0)
        )
        yoffset = 5.25 if has_multiline_below else 0.0

        if show_monomial_lines and show_efold:
            code += f"""add_monomial_legend_entry(legend_handles, legend_labels, handler_map, N_range=({N_min}, {N_max}), yoffset={yoffset})
"""
        elif show_efold:
            code += f"""from matplotlib.patches import Rectangle
dummy_handle = Rectangle((0, 0), 1, 1, facecolor='0.8', edgecolor='0.6')
legend_handles.append(dummy_handle)
legend_labels.append('$V(\\\\phi) \\\\propto \\\\phi^n$\\n(${N_min}\\\\leq\\\\! N_\\\\star\\\\!\\\\leq {N_max}$)')
"""
        else:
            code += f"""from matplotlib.lines import Line2D
dummy_handle = Line2D([], [], color='r', lw=1.2)
legend_handles.append(dummy_handle)
legend_labels.append(r'$V(\\phi) \\propto \\phi^{{n}},\\, n=1, 2/3, 1/3$' + '\\n(${N_min}\\\\leq\\\\! N_\\\\star\\\\!\\\\leq {N_max}$)')
"""

    if show_starobinsky or show_higgs:
        code += """model_handles, model_labels = add_alpha_unity_model_markers(ax, models=models_to_show, return_handles=True)
legend_handles.extend(model_handles)
legend_labels.extend(model_labels)
"""

    if show_alpha_attractor and len(alpha_N_values) > 0:
        yoffset = 0.0 if len(alpha_N_values) > 2 else 5.25
        code += f"""add_alpha_attractor_legend_entry(legend_handles, legend_labels, handler_map, N_star={alpha_N_values}, k={alpha_k}, yoffset={yoffset})
"""

    if show_custom_marker:
        # Escape the label properly for the exported code
        # We need to get the label as it will appear after being rendered by matplotlib
        escaped_label = custom_marker_label
        code += f"""# Add custom marker to legend
plot_handles, plot_labels = ax.get_legend_handles_labels()
for h, l in zip(plot_handles, plot_labels):
    if l == r"{escaped_label}":
        legend_handles.append(h)
        legend_labels.append(l)
        break
"""

    code += f"""add_model_handlers_to_legend(ax, handles=legend_handles, labels=legend_labels,
                             handler_map=handler_map, loc="{legend_loc}",
                             fontsize={legend_fontsize}, handlelength=1.5, ncol=1)

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

# Set axis ticks and spines to foreground
ax.set_axisbelow(False)
for spine in ax.spines.values():
    spine.set_zorder(1000)
ax.xaxis.set_zorder(1000)
ax.yaxis.set_zorder(1000)

ax.set_ylabel("Tensor-to-scalar ratio $r$")
ax.set_xlabel("Scalar spectral index $n_s$")

plt.tight_layout()
plt.savefig("r_ns_plot.pdf", bbox_inches='tight', dpi=400)
plt.show()
"""

# Download buttons (these trigger immediate downloads)
# Buffers were created earlier, before display_plot() cleared the figure
st.sidebar.download_button(
    label="Download as PDF",
    data=pdf_buffer,
    file_name="r_ns_plot.pdf",
    mime="application/pdf",
)

st.sidebar.download_button(
    label="Download as PNG",
    data=png_buffer,
    file_name="r_ns_plot.png",
    mime="image/png",
)

st.sidebar.download_button(
    label="Download Python Script",
    data=code,
    file_name="custom_rns_plot_script.py",
    mime="text/x-python",
)
