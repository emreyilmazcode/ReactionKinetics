"""
Command-line interface for the rxnkinetics toolkit.

Subcommands:
    solve     — Simulate reaction kinetics via ODE
    fit       — Fit rate constant from experimental data
    order     — Determine reaction order by R² comparison
    halflife  — Calculate half-life
    arrhenius — Arrhenius analysis (Ea and A)
    gasfit    — Gas volume kinetics (V∞ estimation)
"""

import argparse
import numpy as np

from rxnkinetics.core import (
    read_csv,
    solve_ode,
    fit_rate_constant,
    determine_order,
    half_life,
    arrhenius_analysis,
    gas_volume_fit,
)
from rxnkinetics.plotting import (
    plot_kinetics,
    plot_fit,
    plot_order_comparison,
    plot_arrhenius,
    plot_gas_kinetics,
)


def cmd_solve(args):
    """Solve kinetics ODE and display results."""
    t_span, X_values = solve_ode(args.order, args.x0, args.k, args.t)
    final = X_values[-1]

    print("=" * 44)
    print("  REACTION KINETICS SIMULATION")
    print("=" * 44)
    print(f"  Reaction Order   : {args.order}")
    print(f"  Initial [X]₀     : {args.x0} M")
    print(f"  Rate Constant k  : {args.k}")
    print(f"  Time Elapsed     : {args.t} s")
    print("-" * 44)
    print(f"  → Remaining [X]  : {final:.6f} M")
    print(f"  → Conversion     : {(1 - final / args.x0) * 100:.2f} %")
    print("=" * 44)

    if not args.no_plot:
        plot_kinetics(t_span, X_values, args.order, args.x0, args.k)


def cmd_fit(args):
    """Fit rate constant from experimental data."""
    t_data, C_data = read_csv(args.file)
    k_opt, k_std, t_fit, C_fit = fit_rate_constant(t_data, C_data, args.order)

    C_pred = np.interp(t_data, t_fit, C_fit)
    SS_res = np.sum((C_data - C_pred) ** 2)
    SS_tot = np.sum((C_data - np.mean(C_data)) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0

    print("=" * 44)
    print("  RATE CONSTANT FITTING RESULT")
    print("=" * 44)
    print(f"  File             : {args.file}")
    print(f"  Reaction Order   : {args.order}")
    print(f"  Data Points      : {len(t_data)}")
    print("-" * 44)
    print(f"  → k              : {k_opt:.6g}")
    print(f"  → k uncertainty  : ± {k_std:.6g}")
    print(f"  → R²             : {R2:.6f}")
    print("=" * 44)

    if not args.no_plot:
        plot_fit(t_data, C_data, t_fit, C_fit, k_opt, args.order)


def cmd_order(args):
    """Determine reaction order from experimental data."""
    t_data, C_data = read_csv(args.file)
    candidates = None
    if args.candidates:
        candidates = [float(x.strip()) for x in args.candidates.split(",")]

    results = determine_order(t_data, C_data, candidates)

    print("=" * 50)
    print("  REACTION ORDER ANALYSIS")
    print("=" * 50)
    print(f"  File         : {args.file}")
    print(f"  Data Points  : {len(t_data)}")
    print("-" * 50)
    print(f"  {'Order':>8}  {'k':>14}  {'R²':>10}")
    print(f"  {'-----':>8}  {'-----------':>14}  {'--------':>10}")
    for n, k_val, r2 in results:
        k_str = f"{k_val:.6g}" if k_val is not None else "failed"
        marker = " ← BEST" if results[0] == (n, k_val, r2) else ""
        print(f"  {n:>8.1f}  {k_str:>14}  {r2:>10.6f}{marker}")
    print("=" * 50)

    best = results[0]
    print(f"\n  Suggested order: {best[0]}, k = {best[1]:.6g}, R² = {best[2]:.6f}")

    if args.plot:
        fits = []
        for n, k_val, r2 in results:
            if k_val is None:
                continue
            _, _, t_f, c_f = fit_rate_constant(t_data, C_data, n)
            fits.append((n, r2, t_f, c_f))
        plot_order_comparison(t_data, C_data, fits)


def cmd_halflife(args):
    """Calculate half-life."""
    if args.order != 1 and args.x0 is None:
        import sys
        sys.exit(f"Error: --x0 is required for order {args.order} reactions.")

    t_half = half_life(args.order, args.k, args.x0)

    print("=" * 44)
    print("  HALF-LIFE CALCULATION")
    print("=" * 44)
    print(f"  Reaction Order   : {args.order}")
    print(f"  Rate Constant k  : {args.k}")
    if args.x0 is not None:
        print(f"  Initial [X]₀     : {args.x0} M")
    print("-" * 44)
    print(f"  → t₁/₂           : {t_half:.6g} s")
    print("=" * 44)

    if args.order == 1:
        print(f"\n  Note: First-order half-life is independent of concentration.")


def cmd_arrhenius(args):
    """Perform Arrhenius analysis."""
    T_data, k_data = read_csv(args.file)
    if len(T_data) < 3:
        print("  Warning: fewer than 3 data points — R² may be unreliable.")
    if np.any(k_data <= 0):
        import sys
        sys.exit("Error: k values must be positive.")

    Ea_kJ, A, R2, x_fit, y_fit = arrhenius_analysis(T_data, k_data)

    print("=" * 50)
    print("  ARRHENIUS ANALYSIS")
    print("=" * 50)
    print(f"  File             : {args.file}")
    print(f"  Data Points      : {len(T_data)}")
    print("-" * 50)
    print(f"  → Ea             : {Ea_kJ:.2f} kJ/mol")
    print(f"  → Ea             : {Ea_kJ * 1000:.1f} J/mol")
    print(f"  → A              : {A:.4g}")
    print(f"  → R²             : {R2:.6f}")
    print("=" * 50)

    if not args.no_plot:
        plot_arrhenius(T_data, k_data, x_fit, y_fit, Ea_kJ, R2)


def cmd_gasfit(args):
    """Fit gas evolution kinetics."""
    t_data, V_data = read_csv(args.file)
    Vinf, k_opt, R2, coeffs, y = gas_volume_fit(t_data, V_data, args.vinf)
    t_half = np.log(2) / k_opt

    print("=" * 50)
    print("  GAS VOLUME KINETICS ANALYSIS")
    print("=" * 50)
    print(f"  File             : {args.file}")
    print(f"  Reaction Order   : {args.order}")
    print(f"  Data Points      : {len(t_data)}")
    print("-" * 50)
    print(f"  → V∞ (estimated) : {Vinf:.4g} mL")
    print(f"  → k              : {k_opt:.6g}")
    print(f"  → R²             : {R2:.6f}")
    print(f"  → t₁/₂           : {t_half:.4g} (time unit)")
    print("=" * 50)

    if not args.no_plot:
        t_fit = np.linspace(t_data.min(), t_data.max(), 300)
        V_model = Vinf * (1 - np.exp(-k_opt * t_fit))
        plot_gas_kinetics(t_data, V_data, t_fit, V_model, Vinf, k_opt, y, coeffs, R2)


def build_parser():
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="rxnkinetics — Chemical Reaction Kinetics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rxnkinetics solve     --order 1 --x0 1.0 --k 0.1 --t 50
  rxnkinetics fit       --file data.csv --order 1
  rxnkinetics order     --file data.csv --plot
  rxnkinetics halflife  --order 1 --k 0.1
  rxnkinetics arrhenius --file temperature.csv
  rxnkinetics gasfit    --file gas_data.csv
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # ── solve ──
    p_solve = sub.add_parser("solve", help="Solve kinetics ODE and plot")
    p_solve.add_argument("--order", type=float, required=True, help="Reaction order")
    p_solve.add_argument("--x0", type=float, required=True, help="Initial concentration (M)")
    p_solve.add_argument("--k", type=float, required=True, help="Rate constant")
    p_solve.add_argument("--t", type=float, required=True, help="Total time (s)")
    p_solve.add_argument("--no-plot", action="store_true", help="Suppress plot")
    p_solve.set_defaults(func=cmd_solve)

    # ── fit ──
    p_fit = sub.add_parser("fit", help="Fit rate constant from data")
    p_fit.add_argument("--file", type=str, required=True, help="CSV file (time, concentration)")
    p_fit.add_argument("--order", type=float, required=True, help="Assumed reaction order")
    p_fit.add_argument("--no-plot", action="store_true", help="Suppress plot")
    p_fit.set_defaults(func=cmd_fit)

    # ── order ──
    p_order = sub.add_parser("order", help="Determine reaction order")
    p_order.add_argument("--file", type=str, required=True, help="CSV file (time, concentration)")
    p_order.add_argument("--candidates", type=str, default=None,
                         help="Orders to test, comma-separated (e.g., 0,1,2)")
    p_order.add_argument("--plot", action="store_true", help="Show comparison plot")
    p_order.set_defaults(func=cmd_order)

    # ── halflife ──
    p_half = sub.add_parser("halflife", help="Calculate half-life")
    p_half.add_argument("--order", type=float, required=True, help="Reaction order")
    p_half.add_argument("--k", type=float, required=True, help="Rate constant")
    p_half.add_argument("--x0", type=float, default=None,
                        help="Initial concentration (required for orders ≠ 1)")
    p_half.set_defaults(func=cmd_halflife)

    # ── arrhenius ──
    p_arr = sub.add_parser("arrhenius", help="Arrhenius analysis (Ea calculation)")
    p_arr.add_argument("--file", type=str, required=True, help="CSV file (T in K, k)")
    p_arr.add_argument("--no-plot", action="store_true", help="Suppress plot")
    p_arr.set_defaults(func=cmd_arrhenius)

    # ── gasfit ──
    p_gas = sub.add_parser("gasfit", help="Gas volume kinetics (V∞ and k)")
    p_gas.add_argument("--file", type=str, required=True, help="CSV file (time, volume)")
    p_gas.add_argument("--order", type=float, default=1, help="Reaction order (default: 1)")
    p_gas.add_argument("--vinf", type=float, default=None,
                       help="Known V∞ (auto-estimated if omitted)")
    p_gas.add_argument("--no-plot", action="store_true", help="Suppress plot")
    p_gas.set_defaults(func=cmd_gasfit)

    return parser


def main():
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
