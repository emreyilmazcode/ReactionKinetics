"""
Core computation functions for chemical reaction kinetics.

Provides ODE solving, curve fitting, reaction order determination,
half-life calculation, and Arrhenius analysis.
"""

import sys
import csv
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def read_csv(filepath):
    """Read a two-column CSV file and return numpy arrays.

    Parameters
    ----------
    filepath : str
        Path to a CSV file with two numeric columns.

    Returns
    -------
    col1, col2 : numpy.ndarray
        Parsed columns as float arrays.

    Raises
    ------
    SystemExit
        If the file has fewer than 2 valid data rows.
    """
    col1, col2 = [], []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            try:
                col1.append(float(row[0].strip()))
                col2.append(float(row[1].strip()))
            except ValueError:
                if i == 0:
                    continue  # skip header
                sys.exit(f"Error: cannot parse '{filepath}' line {i+1}: {row}")
    if len(col1) < 2:
        sys.exit(f"Error: '{filepath}' has insufficient data (at least 2 rows required).")
    return np.array(col1), np.array(col2)


def reaction_ode(X, t, k, n):
    """Rate law ODE: d[X]/dt = -k * [X]^n"""
    return -k * (max(X, 0) ** n)


def solve_ode(order, X0, k, t_final, n_points=500):
    """Solve the kinetics ODE numerically.

    Parameters
    ----------
    order : float
        Reaction order.
    X0 : float
        Initial concentration (M).
    k : float
        Rate constant.
    t_final : float
        End time (s).
    n_points : int
        Number of time points for the solution.

    Returns
    -------
    t_span : numpy.ndarray
        Time array.
    X_values : numpy.ndarray
        Concentration array.
    """
    t_span = np.linspace(0, t_final, n_points)
    solution = odeint(reaction_ode, [X0], t_span, args=(k, order))
    return t_span, solution[:, 0]


def fit_rate_constant(t_data, C_data, order):
    """Fit the rate constant k for a given reaction order.

    Parameters
    ----------
    t_data : numpy.ndarray
        Experimental time values.
    C_data : numpy.ndarray
        Experimental concentration values.
    order : float
        Assumed reaction order.

    Returns
    -------
    k_opt : float
        Optimized rate constant.
    k_std : float
        Standard deviation of k.
    t_fit : numpy.ndarray
        Fitted time array (500 points).
    C_fit : numpy.ndarray
        Fitted concentration array.
    """
    X0 = C_data[0]
    t_max = t_data[-1]

    def model(t_points, k_param):
        t_solve = np.linspace(0, t_max, 500)
        C_solve = odeint(reaction_ode, [X0], t_solve, args=(k_param, order))[:, 0]
        return np.interp(t_points, t_solve, C_solve)

    try:
        popt, pcov = curve_fit(model, t_data, C_data, p0=[0.1], bounds=(0, np.inf))
    except RuntimeError:
        sys.exit("Error: curve fitting did not converge. Check your data and reaction order.")

    k_opt = popt[0]
    k_std = np.sqrt(pcov[0, 0])
    t_fit = np.linspace(0, t_max, 500)
    C_fit = odeint(reaction_ode, [X0], t_fit, args=(k_opt, order))[:, 0]
    return k_opt, k_std, t_fit, C_fit


def determine_order(t_data, C_data, candidates=None):
    """Test candidate reaction orders and rank by R².

    Parameters
    ----------
    t_data : numpy.ndarray
        Experimental time values.
    C_data : numpy.ndarray
        Experimental concentration values.
    candidates : list of float, optional
        Orders to test. Default: [0, 0.5, 1, 1.5, 2, 3].

    Returns
    -------
    results : list of tuple
        Sorted list of (order, k, R²) tuples, best first.
    """
    if candidates is None:
        candidates = [0, 0.5, 1, 1.5, 2, 3]

    SS_tot = np.sum((C_data - np.mean(C_data)) ** 2)
    results = []

    for n in candidates:
        try:
            X0 = C_data[0]
            t_max = t_data[-1]

            def model(t_points, k_param, _n=n):
                t_solve = np.linspace(0, t_max, 500)
                C_solve = odeint(reaction_ode, [X0], t_solve, args=(k_param, _n))[:, 0]
                return np.interp(t_points, t_solve, C_solve)

            popt, _ = curve_fit(model, t_data, C_data, p0=[0.1], bounds=(0, np.inf))
            k_opt = popt[0]
            C_pred = model(t_data, k_opt)
            SS_res = np.sum((C_data - C_pred) ** 2)
            R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
            results.append((n, k_opt, R2))
        except Exception:
            results.append((n, None, -1))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def half_life(order, k, X0=None):
    """Calculate the half-life for a given reaction order.

    Parameters
    ----------
    order : float
        Reaction order.
    k : float
        Rate constant.
    X0 : float, optional
        Initial concentration (required for orders != 1).

    Returns
    -------
    t_half : float
        Half-life value.
    """
    if order == 0:
        if X0 is None:
            sys.exit("Error: --x0 is required for zeroth-order reactions.")
        return X0 / (2 * k)
    elif order == 1:
        return np.log(2) / k
    elif order == 2:
        if X0 is None:
            sys.exit("Error: --x0 is required for second-order reactions.")
        return 1.0 / (k * X0)
    else:
        if X0 is None:
            sys.exit(f"Error: --x0 is required for order {order} reactions.")
        return (2 ** (order - 1) - 1) / (k * (order - 1) * X0 ** (order - 1))


def arrhenius_analysis(T_data, k_data):
    """Perform Arrhenius analysis via linear regression of ln(k) vs 1/T.

    Parameters
    ----------
    T_data : numpy.ndarray
        Temperature values (K).
    k_data : numpy.ndarray
        Rate constant values.

    Returns
    -------
    Ea_kJ : float
        Activation energy in kJ/mol.
    A : float
        Pre-exponential (frequency) factor.
    R2 : float
        Coefficient of determination.
    x_fit : numpy.ndarray
        Fitted 1/T values for plotting.
    y_fit : numpy.ndarray
        Fitted ln(k) values for plotting.
    """
    R = 8.314  # J/(mol·K)
    x = 1.0 / T_data
    y = np.log(k_data)

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    Ea = -slope * R  # J/mol
    A = np.exp(intercept)

    y_pred = np.polyval(coeffs, x)
    SS_res = np.sum((y - y_pred) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)

    return Ea / 1000, A, R2, x_fit, y_fit


def gas_volume_fit(t_data, V_data, Vinf=None):
    """Fit gas evolution kinetics: ln(V∞ - Vt) = ln(V∞) - k*t.

    Parameters
    ----------
    t_data : numpy.ndarray
        Time values.
    V_data : numpy.ndarray
        Volume values (mL).
    Vinf : float, optional
        Known V∞. If None, estimated automatically.

    Returns
    -------
    best_Vinf : float
        Estimated or given V∞.
    k_opt : float
        Optimized rate constant.
    R2 : float
        Coefficient of determination.
    coeffs : numpy.ndarray
        Polynomial coefficients for the linear fit.
    y : numpy.ndarray
        ln(V∞ - Vt) values.
    """
    if Vinf is not None:
        if np.any(V_data >= Vinf):
            sys.exit(f"Error: V∞={Vinf} is less than or equal to some Vt values.")
        y = np.log(Vinf - V_data)
        coeffs = np.polyfit(t_data, y, 1)
        k_opt = -coeffs[0]
        y_pred = np.polyval(coeffs, t_data)
        SS_res = np.sum((y - y_pred) ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)
        R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
        return Vinf, k_opt, R2, coeffs, y

    # Automatic V∞ estimation
    V_max = V_data.max()
    candidates = np.arange(V_max * 1.05, V_max * 3, V_max * 0.01)
    best_R2 = -np.inf
    best_Vinf = None
    best_k = None
    best_coeffs = None

    for Vinf_candidate in candidates:
        try:
            y = np.log(Vinf_candidate - V_data)
            coeffs = np.polyfit(t_data, y, 1)
            y_pred = np.polyval(coeffs, t_data)
            SS_res = np.sum((y - y_pred) ** 2)
            SS_tot = np.sum((y - np.mean(y)) ** 2)
            R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
            if R2 > best_R2:
                best_R2 = R2
                best_Vinf = Vinf_candidate
                best_k = -coeffs[0]
                best_coeffs = coeffs
        except Exception:
            continue

    if best_Vinf is None:
        sys.exit("Error: could not estimate V∞.")

    y = np.log(best_Vinf - V_data)
    return best_Vinf, best_k, best_R2, best_coeffs, y
