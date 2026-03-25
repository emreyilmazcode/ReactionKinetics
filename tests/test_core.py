"""Unit tests for rxnkinetics.core"""

import numpy as np
import pytest
from rxnkinetics.core import solve_ode, half_life, arrhenius_analysis


class TestSolveODE:
    def test_first_order_decay(self):
        t, X = solve_ode(order=1, X0=1.0, k=0.1, t_final=50)
        # At t=50, [X] ≈ e^{-0.1*50} ≈ 0.00674
        assert abs(X[-1] - np.exp(-5)) < 0.01

    def test_zero_order(self):
        t, X = solve_ode(order=0, X0=1.0, k=0.02, t_final=20)
        # At t=20, [X] = 1.0 - 0.02*20 = 0.6
        assert abs(X[-1] - 0.6) < 0.01

    def test_concentration_non_negative(self):
        t, X = solve_ode(order=1, X0=1.0, k=1.0, t_final=100)
        assert np.all(X >= -1e-10)


class TestHalfLife:
    def test_first_order(self):
        t_half = half_life(order=1, k=0.1)
        assert abs(t_half - np.log(2) / 0.1) < 1e-10

    def test_second_order(self):
        t_half = half_life(order=2, k=0.05, X0=1.0)
        assert abs(t_half - 20.0) < 1e-10

    def test_zero_order(self):
        t_half = half_life(order=0, k=0.1, X0=2.0)
        assert abs(t_half - 10.0) < 1e-10


class TestArrhenius:
    def test_known_activation_energy(self):
        R = 8.314
        Ea_true = 50000  # J/mol
        A_true = 1e10
        T = np.array([300, 320, 340, 360, 380])
        k = A_true * np.exp(-Ea_true / (R * T))

        Ea_kJ, A, R2, _, _ = arrhenius_analysis(T, k)
        assert abs(Ea_kJ - 50.0) < 0.1
        assert R2 > 0.999
