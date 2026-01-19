import numpy as np


class IdealGasEOS:
    """
    Ideal-gas EOS for relativistic hydrodynamics.

    P = (Γ - 1) ρ₀ ε
    h = 1 + ε + P/ρ₀
    c_s^2 = Γ P / (ρ₀ h)
    """

    def __init__(self, gamma: float = 1.4):
        """
        Args:
            gamma: Adiabatic index (1 < γ ≤ 2 for relativistic gases).
        """
        if gamma <= 1.0:
            raise ValueError("Adiabatic index must be > 1")

        self.gamma = float(gamma)
        self.gamma_minus_1 = self.gamma - 1.0
        self.name = f"ideal_gas_gamma_{self.gamma}"

    # --- Core thermodynamics ---

    def pressure(self, rho0, eps):
        """P = (Γ - 1) ρ₀ ε"""
        rho0 = np.asarray(rho0, dtype=float)
        eps = np.asarray(eps, dtype=float)
        return self.gamma_minus_1 * rho0 * eps

    def eps_from_rho_p(self, rho0, pressure):
        """ε = P / [(Γ - 1) ρ₀]"""
        rho0 = np.asarray(rho0, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        return pressure / (self.gamma_minus_1 * rho0)

    def enthalpy(self, rho0, pressure, eps):
        """h = 1 + ε + P/ρ₀"""
        rho0 = np.asarray(rho0, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        eps = np.asarray(eps, dtype=float)
        return 1.0 + eps + pressure / rho0

    # Small conveniences used elsewhere
    def enthalpy_from_rho_p(self, rho0, pressure):
        """h using (ρ₀, P) only."""
        eps = self.eps_from_rho_p(rho0, pressure)
        return 1.0 + eps + pressure / np.asarray(rho0, dtype=float)

    def pressure_from_rho_eps(self, rho0, eps):
        """Alias used by some code paths."""
        return self.pressure(rho0, eps)

    # --- Derived speeds etc. ---

    def sound_speed_squared(self, rho0, pressure, eps):
        """
        c_s^2 = Γ P / (ρ₀ h) with h = 1 + ε + P/ρ₀.
        Clipped to [0, 1].
        """
        rho0 = np.asarray(rho0, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        eps = np.asarray(eps, dtype=float)
        h = self.enthalpy(rho0, pressure, eps)
        cs2 = self.gamma * pressure / (rho0 * h)
        return np.clip(cs2, 0.0, 1.0)

    # Optional helpers for Newton-based schemes
    def deps_dp(self, rho0, pressure):
        """dε/dP = 1 / [(Γ - 1) ρ₀]"""
        rho0 = np.asarray(rho0, dtype=float)
        return 1.0 / (self.gamma_minus_1 * rho0)

    def deps_drho(self, rho0, pressure):
        """dε/dρ₀ = -P / [(Γ - 1) ρ₀²]"""
        rho0 = np.asarray(rho0, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        return -pressure / (self.gamma_minus_1 * rho0**2)

    # --- Consistency checks ---
    def validate_state(self, rho0, pressure, eps):
        """
        Returns (is_valid, message) for arrays or scalars.
        """
        rho0 = np.asarray(rho0, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        eps = np.asarray(eps, dtype=float)

        if np.any(rho0 <= 0):
            return False, "Negative or zero density"
        if np.any(pressure < 0):
            return False, "Negative pressure"
        if np.any(eps < 0):
            return False, "Negative specific internal energy"

        p_expected = self.pressure(rho0, eps)
        rel_error = np.abs(pressure - p_expected) / (np.abs(pressure) + 1e-15)
        if np.any(rel_error > 1e-10):
            return False, f"EOS inconsistency, max relative error: {np.max(rel_error)}"

        cs2 = self.sound_speed_squared(rho0, pressure, eps)
        if np.any(cs2 > 1.0):
            return False, "Superluminal sound speed"

        return True, "State is thermodynamically consistent"


class PolytropicEOS:
    """
    Polytropic (barotropic) EOS: P = K ρ₀^Γ
    Commonly used without an evolution equation for ε/τ.

    Relations:
        P = K ρ₀^Γ
        ε = K ρ₀^(Γ-1) / (Γ-1)
        h = 1 + ε + P/ρ₀ = 1 + Γ K ρ₀^(Γ-1) / (Γ-1)
        c_s² = Γ P / (ρ₀ h)

    Derivatives (all analytic):
        dP/dρ₀ = Γ K ρ₀^(Γ-1)
        dε/dρ₀ = K ρ₀^(Γ-2)
        dε/dP  = 1 / (Γ ρ₀)
        dh/dρ₀ = Γ K ρ₀^(Γ-2)
    """

    def __init__(self, K: float, gamma: float):
        if K <= 0:
            raise ValueError("Polytropic constant K must be positive")
        if gamma <= 1.0:
            raise ValueError("Polytropic index must be > 1")

        self.K = float(K)
        self.gamma = float(gamma)
        self.gamma_minus_1 = self.gamma - 1.0
        self.name = f"polytropic_K_{self.K}_gamma_{self.gamma}"

    # =========================================================================
    # Core thermodynamic relations
    # =========================================================================

    def pressure(self, rho0):
        """P(ρ₀) = K ρ₀^Γ"""
        rho0 = np.asarray(rho0, dtype=float)
        return self.K * rho0**self.gamma

    def pressure_from_rho(self, rho0):
        """Alias for pressure(rho0)."""
        return self.pressure(rho0)

    def eps_from_rho(self, rho0):
        """
        ε(ρ₀) = K ρ₀^(Γ-1) / (Γ-1)

        Derivation: From first law dε = P d(1/ρ₀) = -P/ρ₀² dρ₀
        Integrating with P = K ρ₀^Γ gives ε = K ρ₀^(Γ-1)/(Γ-1)
        """
        rho0 = np.asarray(rho0, dtype=float)
        return self.K * rho0**self.gamma_minus_1 / self.gamma_minus_1

    def eps_from_rho_p(self, rho0, pressure):
        """
        For interface compatibility with IdealGasEOS.
        Ignores 'pressure' argument because EOS is barotropic.
        """
        return self.eps_from_rho(rho0)

    def enthalpy(self, rho0):
        """
        h(ρ₀) = 1 + ε + P/ρ₀ = 1 + Γ K ρ₀^(Γ-1) / (Γ-1)

        Simplification:
            h = 1 + K ρ₀^(Γ-1)/(Γ-1) + K ρ₀^Γ/ρ₀
              = 1 + K ρ₀^(Γ-1)/(Γ-1) + K ρ₀^(Γ-1)
              = 1 + K ρ₀^(Γ-1) [1/(Γ-1) + 1]
              = 1 + K ρ₀^(Γ-1) Γ/(Γ-1)
        """
        rho0 = np.asarray(rho0, dtype=float)
        return 1.0 + self.gamma * self.K * rho0**self.gamma_minus_1 / self.gamma_minus_1

    def sound_speed_squared(self, rho0, pressure=None, eps=None):
        """
        c_s² = Γ P / (ρ₀ h) = Γ K ρ₀^(Γ-1) / h

        Compatible interface: ignores pressure and eps since barotropic.
        Result is clipped to [0, 1] for relativistic consistency.
        """
        rho0 = np.asarray(rho0, dtype=float)
        h = self.enthalpy(rho0)
        cs2 = self.gamma * self.K * rho0**self.gamma_minus_1 / h
        return np.clip(cs2, 0.0, 1.0)

    # =========================================================================
    # Compatibility methods (interface consistency with IdealGasEOS)
    # =========================================================================

    def enthalpy_from_rho_p(self, rho0, pressure):
        """Alias for enthalpy - ignores pressure since barotropic."""
        return self.enthalpy(rho0)

    def pressure_from_rho_eps(self, rho0, eps):
        """P(ρ₀) - ignores eps since barotropic."""
        return self.pressure(rho0)

    # =========================================================================
    # Analytic derivatives (for Newton-Raphson and Jacobian computations)
    # =========================================================================

    def dp_drho(self, rho0):
        """
        dP/dρ₀ = Γ K ρ₀^(Γ-1)

        Derivation: P = K ρ₀^Γ  →  dP/dρ₀ = Γ K ρ₀^(Γ-1)
        """
        rho0 = np.asarray(rho0, dtype=float)
        return self.gamma * self.K * rho0**self.gamma_minus_1

    def deps_drho(self, rho0, pressure=None):
        """
        dε/dρ₀ = K ρ₀^(Γ-2)

        Derivation: ε = K ρ₀^(Γ-1)/(Γ-1)
                    dε/dρ₀ = K (Γ-1) ρ₀^(Γ-2) / (Γ-1) = K ρ₀^(Γ-2)

        Alternative form using P:
            Since K = P/ρ₀^Γ  →  dε/dρ₀ = P/ρ₀²

        Note: For barotropic EOS, this is the total derivative (not partial).
              The 'pressure' argument is ignored but included for interface
              compatibility with IdealGasEOS.
        """
        rho0 = np.asarray(rho0, dtype=float)
        return self.K * rho0**(self.gamma - 2.0)

    def deps_dp(self, rho0, pressure=None):
        """
        dε/dP = 1 / (Γ ρ₀)

        Derivation using chain rule:
            dε/dP = (dε/dρ₀) / (dP/dρ₀)
                  = [K ρ₀^(Γ-2)] / [Γ K ρ₀^(Γ-1)]
                  = 1 / (Γ ρ₀)

        Note: The 'pressure' argument is ignored but included for interface
              compatibility with IdealGasEOS.
        """
        rho0 = np.asarray(rho0, dtype=float)
        return 1.0 / (self.gamma * rho0)

    def dh_drho(self, rho0):
        """
        dh/dρ₀ = Γ K ρ₀^(Γ-2)

        Derivation: h = 1 + Γ K ρ₀^(Γ-1)/(Γ-1)
                    dh/dρ₀ = Γ K (Γ-1) ρ₀^(Γ-2) / (Γ-1) = Γ K ρ₀^(Γ-2)
        """
        rho0 = np.asarray(rho0, dtype=float)
        return self.gamma * self.K * rho0**(self.gamma - 2.0)

    def dcs2_drho(self, rho0):
        """
        d(c_s²)/dρ₀

        Derivation: c_s² = Γ K ρ₀^(Γ-1) / h
        Using quotient rule with h = 1 + Γ K ρ₀^(Γ-1)/(Γ-1):

        Let A = Γ K ρ₀^(Γ-1), so c_s² = A/h
        dA/dρ₀ = Γ K (Γ-1) ρ₀^(Γ-2)
        dh/dρ₀ = Γ K ρ₀^(Γ-2)

        d(c_s²)/dρ₀ = [h dA/dρ₀ - A dh/dρ₀] / h²
                    = [Γ(Γ-1) K ρ₀^(Γ-2) h - Γ K ρ₀^(Γ-1) · Γ K ρ₀^(Γ-2)] / h²
                    = Γ K ρ₀^(Γ-2) [(Γ-1)h - Γ K ρ₀^(Γ-1)] / h²

        Since h(Γ-1) = (Γ-1) + Γ K ρ₀^(Γ-1):
            (Γ-1)h - Γ K ρ₀^(Γ-1) = (Γ-1)

        Final result: d(c_s²)/dρ₀ = Γ K (Γ-1) ρ₀^(Γ-2) / h²

        Note: This is the derivative of the unclipped c_s². The clipping in
        sound_speed_squared() is not differentiable at the boundaries.
        """
        rho0 = np.asarray(rho0, dtype=float)
        h = self.enthalpy(rho0)
        return self.gamma * self.K * self.gamma_minus_1 * rho0**(self.gamma - 2.0) / h**2

    # =========================================================================
    # Inverse relations
    # =========================================================================

    def rho_from_pressure(self, pressure):
        """
        ρ₀(P) = (P/K)^(1/Γ)

        Inverse of P = K ρ₀^Γ
        """
        pressure = np.asarray(pressure, dtype=float)
        return (pressure / self.K)**(1.0 / self.gamma)

    def rho_from_enthalpy(self, h):
        """
        ρ₀(h) = [(h-1)(Γ-1)/(Γ K)]^(1/(Γ-1))

        Inverse of h = 1 + Γ K ρ₀^(Γ-1)/(Γ-1)
        """
        h = np.asarray(h, dtype=float)
        return ((h - 1.0) * self.gamma_minus_1 / (self.gamma * self.K))**(1.0 / self.gamma_minus_1)

    # =========================================================================
    # Consistency checks
    # =========================================================================

    def validate_state(self, rho0, pressure=None, eps=None):
        """
        Validates thermodynamic state consistency.

        For barotropic EOS, pressure and eps are computed from rho0,
        so the check verifies internal consistency if provided.

        Returns:
            (is_valid, message): Tuple with validation result.
        """
        rho0 = np.asarray(rho0, dtype=float)

        # Check density
        if np.any(rho0 <= 0):
            return False, "Negative or zero density"

        # Compute expected values
        p_expected = self.pressure(rho0)
        eps_expected = self.eps_from_rho(rho0)

        # Validate pressure if provided
        if pressure is not None:
            pressure = np.asarray(pressure, dtype=float)
            if np.any(pressure < 0):
                return False, "Negative pressure"
            rel_error_p = np.abs(pressure - p_expected) / (np.abs(p_expected) + 1e-15)
            if np.any(rel_error_p > 1e-10):
                return False, f"Pressure inconsistent with barotropic EOS, max relative error: {np.max(rel_error_p)}"

        # Validate eps if provided
        if eps is not None:
            eps = np.asarray(eps, dtype=float)
            if np.any(eps < 0):
                return False, "Negative specific internal energy"
            rel_error_eps = np.abs(eps - eps_expected) / (np.abs(eps_expected) + 1e-15)
            if np.any(rel_error_eps > 1e-10):
                return False, f"Epsilon inconsistent with barotropic EOS, max relative error: {np.max(rel_error_eps)}"

        # Check sound speed
        cs2 = self.sound_speed_squared(rho0)
        if np.any(cs2 > 1.0):
            return False, "Superluminal sound speed"

        return True, "State is thermodynamically consistent"


# =============================================================================
# Factory function
# =============================================================================

def create_eos(eos_type: str = "ideal", **kwargs):
    """
    Factory for EOS objects.

    Args:
        eos_type: "ideal" or "polytropic"
        **kwargs: parameters for the chosen EOS
            - For "ideal": gamma (default 1.4)
            - For "polytropic": K (default 1.0), gamma (default 2.0)

    Returns:
        An EOS instance with the standard interface.

    Examples:
        >>> eos = create_eos("ideal", gamma=5/3)
        >>> eos = create_eos("polytropic", K=100, gamma=2)
    """
    if eos_type == "ideal":
        return IdealGasEOS(kwargs.get("gamma", 1.4))
    elif eos_type == "polytropic":
        return PolytropicEOS(kwargs.get("K", 1.0), kwargs.get("gamma", 2.0))
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")


# =============================================================================
# Unit tests for derivative verification
# =============================================================================

def _test_single_derivative(name, analytic, numeric, tol=1e-6):
    """Helper to test a single derivative."""
    rel_error = np.abs(analytic - numeric) / (np.abs(analytic) + 1e-15)
    max_err = np.max(rel_error)
    passed = max_err < tol
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {name}: max rel error = {max_err:.2e} {status}")
    return passed


def test_polytropic_derivatives(K=100.0, gamma=2.0, verbose=True):
    """
    Verify analytic derivatives against numerical finite differences.

    Args:
        K: Polytropic constant
        gamma: Adiabatic index
        verbose: Print detailed output

    Returns:
        bool: True if all tests pass
    """
    if verbose:
        print("=" * 60)
        print(f"Testing PolytropicEOS: K = {K}, Γ = {gamma}")
        print("=" * 60)

    eos = PolytropicEOS(K, gamma)

    # Test points (avoid very small rho for numerical stability with gamma < 2)
    rho_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    h_fd = 1e-8  # Finite difference step

    all_passed = True

    # Test dP/dρ₀
    dp_drho_analytic = eos.dp_drho(rho_test)
    dp_drho_numeric = (eos.pressure(rho_test + h_fd) - eos.pressure(rho_test - h_fd)) / (2 * h_fd)
    all_passed &= _test_single_derivative("dP/dρ₀", dp_drho_analytic, dp_drho_numeric)

    # Test dε/dρ₀
    deps_drho_analytic = eos.deps_drho(rho_test)
    deps_drho_numeric = (eos.eps_from_rho(rho_test + h_fd) - eos.eps_from_rho(rho_test - h_fd)) / (2 * h_fd)
    all_passed &= _test_single_derivative("dε/dρ₀", deps_drho_analytic, deps_drho_numeric, tol=1e-5)

    # Test dε/dP (via chain rule)
    deps_dp_analytic = eos.deps_dp(rho_test)
    deps_dp_chain = deps_drho_analytic / dp_drho_analytic
    all_passed &= _test_single_derivative("dε/dP", deps_dp_analytic, deps_dp_chain, tol=1e-12)

    # Test dh/dρ₀ - slightly higher tolerance for extreme gamma values
    dh_drho_analytic = eos.dh_drho(rho_test)
    dh_drho_numeric = (eos.enthalpy(rho_test + h_fd) - eos.enthalpy(rho_test - h_fd)) / (2 * h_fd)
    all_passed &= _test_single_derivative("dh/dρ₀", dh_drho_analytic, dh_drho_numeric, tol=1e-5)

    # Test d(c_s²)/dρ₀ (unclipped) - higher tolerance due to compound numerical error
    dcs2_drho_analytic = eos.dcs2_drho(rho_test)
    def cs2_unclipped(rho):
        h_val = eos.enthalpy(rho)
        return eos.gamma * eos.K * rho**eos.gamma_minus_1 / h_val
    dcs2_drho_numeric = (cs2_unclipped(rho_test + h_fd) - cs2_unclipped(rho_test - h_fd)) / (2 * h_fd)
    all_passed &= _test_single_derivative("d(c_s²)/dρ₀", dcs2_drho_analytic, dcs2_drho_numeric, tol=1e-4)

    # Test inverse functions
    p_test = eos.pressure(rho_test)
    rho_recovered = eos.rho_from_pressure(p_test)
    all_passed &= _test_single_derivative("ρ₀(P) inverse", rho_recovered, rho_test, tol=1e-12)

    h_test = eos.enthalpy(rho_test)
    rho_from_h = eos.rho_from_enthalpy(h_test)
    all_passed &= _test_single_derivative("ρ₀(h) inverse", rho_from_h, rho_test, tol=1e-12)

    # Test validate_state
    valid, msg = eos.validate_state(rho_test)
    if not valid:
        print(f"  validate_state: ✗ FAILED - {msg}")
        all_passed = False
    else:
        print(f"  validate_state: ✓ PASSED")

    if verbose:
        print("=" * 60)
        print("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗")
        print("=" * 60)

    return all_passed


def test_all_eos():
    """Run comprehensive tests for multiple EOS configurations."""
    print("\n" + "=" * 70)
    print(" COMPREHENSIVE EOS TEST SUITE")
    print("=" * 70)

    all_passed = True

    # Test various gamma values for polytropic EOS
    test_configs = [
        (1.0, 2.0, "Stiff EOS (Γ=2)"),
        (100.0, 5/3, "Monatomic gas (Γ=5/3)"),
        (50.0, 4/3, "Relativistic gas (Γ=4/3)"),
        (1.0, 1.4, "Diatomic gas (Γ=1.4)"),
        (0.01, 1.1, "Soft EOS (Γ=1.1)"),
    ]

    for K, gamma, name in test_configs:
        print(f"\n>>> {name}: K={K}, Γ={gamma:.4f}")
        passed = test_polytropic_derivatives(K, gamma, verbose=False)
        all_passed &= passed
        if passed:
            print("    All derivatives verified ✓")
        else:
            print("    Some tests failed ✗")

    # Test IdealGasEOS
    print("\n>>> IdealGasEOS tests")
    for gamma in [1.4, 5/3, 4/3]:
        eos = IdealGasEOS(gamma)
        rho = np.array([0.5, 1.0, 2.0])
        p = np.array([0.1, 0.5, 1.0])
        eps = eos.eps_from_rho_p(rho, p)
        valid, msg = eos.validate_state(rho, p, eps)
        status = "✓" if valid else "✗"
        print(f"    Γ={gamma:.4f}: {status} {msg}")
        all_passed &= valid

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL TESTS PASSED ✓")
    else:
        print(" SOME TESTS FAILED ✗")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    test_all_eos()

def create_eos(eos_type: str = "ideal", **kwargs):
    """
    Factory for EOS objects.

    Args:
        eos_type: "ideal", "polytropic", or "tabulated"
        **kwargs: parameters for the chosen EOS

    Returns:
        An EOS instance with the standard interface.
    """
    if eos_type == "ideal":
        return IdealGasEOS(kwargs.get("gamma", 1.4))
    elif eos_type == "polytropic":
        return PolytropicEOS(kwargs.get("K", 1.0), kwargs.get("gamma", 2.0))
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")


# =============================================================================
# Unit tests for derivative verification
# =============================================================================

def test_polytropic_derivatives():
    """
    Verify analytic derivatives against numerical finite differences.
    """
    print("=" * 60)
    print("Testing PolytropicEOS analytic derivatives")
    print("=" * 60)

    # Test parameters
    K = 100.0
    gamma = 2.0
    eos = PolytropicEOS(K, gamma)

    # Test points
    rho_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    h = 1e-8  # Finite difference step

    print(f"\nEOS: K = {K}, Γ = {gamma}")
    print(f"Test densities: {rho_test}")
    print(f"Finite difference step: h = {h}")

    all_passed = True

    # Test dP/dρ₀
    print("\n--- Testing dP/dρ₀ ---")
    dp_drho_analytic = eos.dp_drho(rho_test)
    dp_drho_numeric = (eos.pressure(rho_test + h) - eos.pressure(rho_test - h)) / (2 * h)
    rel_error = np.abs(dp_drho_analytic - dp_drho_numeric) / (np.abs(dp_drho_analytic) + 1e-15)
    print(f"  Analytic:  {dp_drho_analytic}")
    print(f"  Numeric:   {dp_drho_numeric}")
    print(f"  Rel error: {rel_error}")
    if np.all(rel_error < 1e-6):
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        all_passed = False

    # Test dε/dρ₀
    print("\n--- Testing dε/dρ₀ ---")
    deps_drho_analytic = eos.deps_drho(rho_test)
    deps_drho_numeric = (eos.eps_from_rho(rho_test + h) - eos.eps_from_rho(rho_test - h)) / (2 * h)
    rel_error = np.abs(deps_drho_analytic - deps_drho_numeric) / (np.abs(deps_drho_analytic) + 1e-15)
    print(f"  Analytic:  {deps_drho_analytic}")
    print(f"  Numeric:   {deps_drho_numeric}")
    print(f"  Rel error: {rel_error}")
    if np.all(rel_error < 1e-6):
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        all_passed = False

    # Test dε/dP (via chain rule verification)
    print("\n--- Testing dε/dP ---")
    deps_dp_analytic = eos.deps_dp(rho_test)
    # dε/dP = (dε/dρ₀) / (dP/dρ₀)
    deps_dp_chain = deps_drho_analytic / dp_drho_analytic
    rel_error = np.abs(deps_dp_analytic - deps_dp_chain) / (np.abs(deps_dp_analytic) + 1e-15)
    print(f"  Analytic:       {deps_dp_analytic}")
    print(f"  Chain rule:     {deps_dp_chain}")
    print(f"  Expected 1/(Γρ): {1.0 / (gamma * rho_test)}")
    print(f"  Rel error:      {rel_error}")
    if np.all(rel_error < 1e-12):
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        all_passed = False

    # Test dh/dρ₀
    print("\n--- Testing dh/dρ₀ ---")
    dh_drho_analytic = eos.dh_drho(rho_test)
    dh_drho_numeric = (eos.enthalpy(rho_test + h) - eos.enthalpy(rho_test - h)) / (2 * h)
    rel_error = np.abs(dh_drho_analytic - dh_drho_numeric) / (np.abs(dh_drho_analytic) + 1e-15)
    print(f"  Analytic:  {dh_drho_analytic}")
    print(f"  Numeric:   {dh_drho_numeric}")
    print(f"  Rel error: {rel_error}")
    if np.all(rel_error < 1e-6):
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        all_passed = False

    # Test d(c_s²)/dρ₀ (using unclipped version for proper derivative comparison)
    print("\n--- Testing d(c_s²)/dρ₀ ---")
    dcs2_drho_analytic = eos.dcs2_drho(rho_test)
    # Compute unclipped c_s² for numerical derivative
    def cs2_unclipped(rho):
        h_val = eos.enthalpy(rho)
        return eos.gamma * eos.K * rho**eos.gamma_minus_1 / h_val
    dcs2_drho_numeric = (cs2_unclipped(rho_test + h) - cs2_unclipped(rho_test - h)) / (2 * h)
    rel_error = np.abs(dcs2_drho_analytic - dcs2_drho_numeric) / (np.abs(dcs2_drho_analytic) + 1e-15)
    print(f"  Analytic:  {dcs2_drho_analytic}")
    print(f"  Numeric:   {dcs2_drho_numeric}")
    print(f"  Rel error: {rel_error}")
    if np.all(rel_error < 1e-5):
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        all_passed = False

    # Test inverse functions
    print("\n--- Testing inverse functions ---")
    p_test = eos.pressure(rho_test)
    rho_recovered = eos.rho_from_pressure(p_test)
    rel_error_rho = np.abs(rho_recovered - rho_test) / rho_test
    print(f"  ρ₀ original:   {rho_test}")
    print(f"  ρ₀ recovered:  {rho_recovered}")
    print(f"  Rel error:     {rel_error_rho}")
    if np.all(rel_error_rho < 1e-12):
        print("  ✓ PASSED (rho_from_pressure)")
    else:
        print("  ✗ FAILED (rho_from_pressure)")
        all_passed = False

    h_test = eos.enthalpy(rho_test)
    rho_from_h = eos.rho_from_enthalpy(h_test)
    rel_error_h = np.abs(rho_from_h - rho_test) / rho_test
    print(f"  ρ₀ from h:     {rho_from_h}")
    print(f"  Rel error:     {rel_error_h}")
    if np.all(rel_error_h < 1e-12):
        print("  ✓ PASSED (rho_from_enthalpy)")
    else:
        print("  ✗ FAILED (rho_from_enthalpy)")
        all_passed = False

    # Test validate_state
    print("\n--- Testing validate_state ---")
    valid, msg = eos.validate_state(rho_test)
    print(f"  Result: {valid}, '{msg}'")
    if valid:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


def create_eos(eos_type: str = "ideal", **kwargs):
    """
    Factory for EOS objects.

    Args:
        eos_type: "ideal", "polytropic", or "tabulated"
        **kwargs: parameters for the chosen EOS

    Returns:
        An EOS instance with the standard interface.
    """
    if eos_type == "ideal":
        return IdealGasEOS(kwargs.get("gamma", 1.4))
    elif eos_type == "polytropic":
        return PolytropicEOS(kwargs.get("K", 1.0), kwargs.get("gamma", 2.0))
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")


