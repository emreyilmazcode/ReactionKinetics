# rxnkinetics

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python CLI toolkit for **chemical reaction kinetics** analysis. Solve rate equations, fit experimental data, determine reaction orders, and perform Arrhenius analysis — all from the command line.

## Features

| Command | Description |
|---------|-------------|
| `solve` | Simulate concentration–time profiles by solving **d[X]/dt = −k[X]ⁿ** numerically |
| `fit` | Determine the rate constant **k** from experimental data via nonlinear least-squares fitting |
| `order` | Test multiple candidate orders and rank by **R²** to identify the best-fit reaction order |
| `halflife` | Calculate **t₁/₂** for any reaction order (0, 1, 2, fractional) |
| `arrhenius` | Compute activation energy **Eₐ** and pre-exponential factor **A** from temperature-dependent rate data |
| `gasfit` | Analyze gas evolution kinetics — estimate **V∞** and extract **k** via linearization |

## Installation

```bash
git clone https://github.com/emreyilmazcode/ReactionKinetics.git
cd ReactionKinetics
pip install -e .
```

After installation, `rxnkinetics` is available as a command:

```bash
rxnkinetics solve --order 1 --x0 1.0 --k 0.1 --t 50
```

Or run directly without installing:

```bash
python -m rxnkinetics.cli solve --order 1 --x0 1.0 --k 0.1 --t 50
```

## Quick Start

### Simulate a first-order decay

```bash
rxnkinetics solve --order 1 --x0 1.0 --k 0.1 --t 50
```

```
============================================
  REACTION KINETICS SIMULATION
============================================
  Reaction Order   : 1.0
  Initial [X]₀     : 1.0 M
  Rate Constant k  : 0.1
  Time Elapsed     : 50.0 s
--------------------------------------------
  → Remaining [X]  : 0.006738 M
  → Conversion     : 99.33 %
============================================
```

### Fit rate constant from experimental data

```bash
rxnkinetics fit --file examples/first_order.csv --order 1
```

### Determine reaction order automatically

```bash
rxnkinetics order --file examples/first_order.csv --plot
```

### Calculate half-life

```bash
rxnkinetics halflife --order 1 --k 0.1
rxnkinetics halflife --order 2 --k 0.05 --x0 1.0
```

### Arrhenius analysis

```bash
rxnkinetics arrhenius --file examples/arrhenius.csv
```

### Gas volume kinetics

```bash
rxnkinetics gasfit --file examples/gas_volume.csv
```

## Input Data Format

All commands expecting a `--file` argument use two-column CSV files:

| Command | Column 1 | Column 2 |
|---------|----------|----------|
| `fit`, `order` | Time (s) | Concentration (M) |
| `arrhenius` | Temperature (K) | Rate constant (k) |
| `gasfit` | Time | Volume (mL) |

Header rows are automatically detected and skipped.

## Theory

### Rate Law Integration

The tool numerically integrates the general rate law:

$$\frac{d[X]}{dt} = -k[X]^n$$

using `scipy.integrate.odeint`. Rate constant fitting uses `scipy.optimize.curve_fit` with the ODE solution as the model function.

### Arrhenius Equation

Temperature dependence is analyzed via the linearized Arrhenius equation:

$$\ln k = \ln A - \frac{E_a}{RT}$$

A linear regression of ln(k) vs 1/T yields the activation energy **Eₐ** and pre-exponential factor **A**.

### Half-Life Expressions

| Order | t₁/₂ |
|-------|-------|
| 0 | [X]₀ / 2k |
| 1 | ln(2) / k |
| 2 | 1 / (k[X]₀) |
| n | (2ⁿ⁻¹ − 1) / [k(n−1)[X]₀ⁿ⁻¹] |

## Project Structure

```
ReactionKinetics/
├── rxnkinetics/
│   ├── __init__.py      # Package metadata
│   ├── core.py          # ODE solver, fitting, analysis functions
│   ├── cli.py           # Command-line interface
│   └── plotting.py      # Matplotlib visualization
├── examples/            # Sample CSV datasets
├── tests/               # Unit tests
├── setup.py             # Package configuration
├── requirements.txt
├── LICENSE
└── README.md
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4

## License

[MIT](LICENSE)
