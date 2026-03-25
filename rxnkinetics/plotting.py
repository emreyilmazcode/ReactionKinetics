"""
Plotting utilities for reaction kinetics visualization.

All plot functions use matplotlib and follow a consistent style.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_kinetics(t_span, X_values, order, X0, k):
    """Plot concentration vs time for a kinetics simulation."""
    plt.figure(figsize=(8, 5))
    plt.plot(t_span, X_values, label=f"Order {order}", color="royalblue", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration [X] (M)")
    plt.title(f"Reaction Kinetics ([X]₀ = {X0} M, k = {k})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fit(t_data, C_data, t_fit, C_fit, k_opt, order):
    """Plot experimental data with the fitted curve."""
    plt.figure(figsize=(8, 5))
    plt.scatter(t_data, C_data, color="crimson", label="Experimental Data", zorder=5)
    plt.plot(t_fit, C_fit, color="royalblue", linewidth=2,
             label=f"Fit (k={k_opt:.4g}, n={order})")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration [X] (M)")
    plt.title(f"Rate Constant Fitting (Order {order})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_order_comparison(t_data, C_data, fits):
    """Plot experimental data with multiple order fits.

    Parameters
    ----------
    t_data : numpy.ndarray
        Experimental time values.
    C_data : numpy.ndarray
        Experimental concentration values.
    fits : list of tuple
        Each tuple: (order, R2, t_fit, C_fit).
    """
    colors = ["royalblue", "crimson", "forestgreen", "darkorange", "purple", "brown"]
    plt.figure(figsize=(10, 6))
    plt.scatter(t_data, C_data, color="black", label="Experimental Data", zorder=5, s=40)
    for i, (n, r2, t_f, c_f) in enumerate(fits):
        plt.plot(t_f, c_f, color=colors[i % len(colors)], linewidth=1.5,
                 label=f"n={n} (R²={r2:.4f})")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration [X] (M)")
    plt.title("Reaction Order Comparison")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_arrhenius(T_data, k_data, x_fit, y_fit, Ea_kJ, R2):
    """Plot Arrhenius diagram: ln(k) vs 1/T."""
    plt.figure(figsize=(8, 5))
    plt.scatter(1.0 / T_data, np.log(k_data), color="crimson",
                label="Experimental Data", zorder=5)
    plt.plot(x_fit, y_fit, color="royalblue", linewidth=2,
             label=f"Regression (Ea={Ea_kJ:.1f} kJ/mol)")
    plt.xlabel("1/T (1/K)")
    plt.ylabel("ln(k)")
    plt.title("Arrhenius Plot")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gas_kinetics(t_data, V_data, t_fit, V_model, Vinf, k_opt, y, coeffs, R2):
    """Plot gas volume kinetics: V vs t and linearization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: V vs t
    ax1.scatter(t_data, V_data, color="crimson", label="Experimental Data", zorder=5)
    ax1.plot(t_fit, V_model, color="royalblue", linewidth=2,
             label=f"Model (V∞={Vinf:.3g}, k={k_opt:.4g})")
    ax1.axhline(y=Vinf, color="gray", linestyle="--", alpha=0.7,
                label=f"V∞ = {Vinf:.3g} mL")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("V (mL)")
    ax1.set_title("Volume vs Time")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Right: linearization
    t_line = np.linspace(t_data.min(), t_data.max(), 100)
    y_line = np.polyval(coeffs, t_line)
    ax2.scatter(t_data, y, color="crimson", label="ln(V∞ − Vt)", zorder=5)
    ax2.plot(t_line, y_line, color="royalblue", linewidth=2,
             label=f"Linear Fit (R²={R2:.4f})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("ln(V∞ − Vt)")
    ax2.set_title("Linearization Plot")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()
