# Code to calculate r and ns predictions for a given potential
# -> This code has not been stress-tested and it may not always be correct; double check important results!
# Used to calculate the predictions of polynomial alpha attractors (settings below)
# Requires JAX

# Settings used for the polynomial alpha attractors
# k=1
# mu_min = 0.001
# mu_max = 1e3
# mu_N = 100
# mu_values = jnp.sort(jnp.unique(jnp.array(list(jnp.logspace(jnp.log10(mu_min), jnp.log10(mu_max), mu_N))+ list(jnp.linspace(mu_min, mu_max, mu_N)))))

# k=2
# mu_min = 0.04
# mu_max = 50
# mu_N = 100
# mu_values = jnp.sort(jnp.unique(jnp.array(list(jnp.logspace(jnp.log10(mu_min), jnp.log10(mu_max), mu_N)) + list(jnp.linspace(mu_min, mu_max, mu_N)))))

# k=3
# mu_min = 0.1
# mu_max = 50
# mu_N = 100
# mu_values = jnp.sort(jnp.unique(jnp.array(list(jnp.logspace(jnp.log10(mu_min), jnp.log10(mu_max), mu_N)) + list(jnp.linspace(mu_min, mu_max, mu_N)))))

# k=4
# mu_min = 0.1
# mu_max = 30
# mu_N = 100
# mu_values = jnp.sort(jnp.unique(jnp.array(list(jnp.logspace(jnp.log10(mu_min), jnp.log10(mu_max), mu_N)) + list(jnp.linspace(mu_min, mu_max, mu_N)))))

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import jax
import jax.numpy as jnp
from jax import grad, vmap
import scipy.optimize as opt
from functools import partial

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
Mpl = 1.0


# ---------------------------------------------------------------------
# Slow-roll parameters (vectorized-friendly)
# ---------------------------------------------------------------------
def epsilon_V(phi, V, dV):
    return 0.5 * Mpl**2 * (dV(phi) / V(phi)) ** 2


def eta_V(phi, V, d2V):
    return Mpl**2 * (d2V(phi) / V(phi))


def find_phi_end(phi_guess, V, dV):
    f = lambda phi: epsilon_V(phi, V, dV) - 1.0
    root = opt.root_scalar(
        f, bracket=[0.01, phi_guess], method="brentq"
    )  # bracket=[0.1, phi_guess]
    return root.root


def compute_N(phi, phi_end, V, dV):
    integrand = lambda p: V(p) / dV(p)
    ps = jnp.linspace(phi_end, phi, 2000)
    vals = jax.vmap(integrand)(ps)
    return jnp.trapezoid(vals, ps) / Mpl**2


def find_phi_N(N_target, phi_end, phi_guess, V, dV):
    f = lambda phi: float(compute_N(phi, phi_end, V, dV)) - N_target
    root = opt.root_scalar(f, bracket=[phi_end + 0.01, phi_guess], method="brentq")
    return root.root


def inflation_predictions_single(V_func, N_target, phi_guess=50.0):
    dV = jax.grad(V_func)
    d2V = jax.grad(dV)

    phi_end = find_phi_end(phi_guess, V_func, dV)
    phi_N = find_phi_N(N_target, phi_end, phi_guess, V_func, dV)

    eps = epsilon_V(phi_N, V_func, dV)
    eta = eta_V(phi_N, V_func, d2V)

    ns = 1 - 6 * eps + 2 * eta
    r = 16 * eps

    return jnp.array([ns, r])


def inflation_predictions_paa(k, mu, N_target, phi_guess=50.0):  # 15
    """Function tailored for polynomial alpha attractors"""
    V_func = lambda phi: 1e-10 * (jnp.abs(phi) ** k) / (mu**k + (jnp.abs(phi) ** k))
    dV = jax.grad(V_func)
    d2V = jax.grad(dV)

    phi_end = find_phi_end(phi_guess, V_func, dV)
    phi_N = find_phi_N(N_target, phi_end, phi_guess, V_func, dV)

    eps = epsilon_V(phi_N, V_func, dV)
    eta = eta_V(phi_N, V_func, d2V)

    ns = 1 - 6 * eps + 2 * eta
    r = 16 * eps

    return jnp.array([ns, r])


def compute_predictions_for_params(k, N_star, mu_values):
    """Compute predictions for all mu values given k and N_star."""
    results = []
    for mu in mu_values:
        try:
            res = inflation_predictions_single(k, float(mu), N_star)
            results.append(res)
        except Exception as e:
            print(f"Failed k={k}, mu={mu}, N_star={N_star}: {e}")
            results.append(jnp.array([jnp.nan, jnp.nan]))
    return jnp.stack(results)
