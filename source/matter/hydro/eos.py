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
        if gamma > 2.0:
            print(f"Warning: γ = {gamma} > 2 may be unphysical for relativistic gas")

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

    def temperature(self, rho0, pressure, eps, k_boltzmann: float = 1.0, mass_unit: float = 1.0):
        """
        Very rough ideal-gas temperature: P = ρ₀ T / (μ m_u); assume μ m_u = mass_unit.
        """
        rho0 = np.asarray(rho0, dtype=float)
        pressure = np.asarray(pressure, dtype=float)
        return pressure * mass_unit / (rho0 * k_boltzmann)

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

    # Core relations
    def pressure(self, rho0):
        """P(ρ₀)"""
        rho0 = np.asarray(rho0, dtype=float)
        return self.K * rho0**self.gamma

    def pressure_from_rho(self, rho0):
        """Alias."""
        return self.pressure(rho0)

    def eps_from_rho(self, rho0):
        """ε(ρ₀) = K ρ₀^(Γ-1)/(Γ-1)"""
        rho0 = np.asarray(rho0, dtype=float)
        return self.K * rho0**(self.gamma_minus_1) / self.gamma_minus_1

    def eps_from_rho_p(self, rho0, pressure):
        """
        For interface compatibility with ideal gas; ignores 'pressure'
        because barotropic.
        """
        return self.eps_from_rho(rho0)

    def enthalpy(self, rho0):
        """
        h(ρ₀) = 1 + ε + P/ρ₀ = 1 + Γ K ρ₀^(Γ-1)/(Γ-1)
        """
        rho0 = np.asarray(rho0, dtype=float)
        return 1.0 + self.gamma * self.K * rho0**(self.gamma_minus_1) / self.gamma_minus_1

    def sound_speed_squared(self, rho0, pressure=None, eps=None):
        """
        c_s^2 = Γ P / (ρ₀ h) = Γ K ρ₀^(Γ-1) / h

        Compatible interface: ignores pressure and eps since barotropic.
        """
        rho0 = np.asarray(rho0, dtype=float)
        h = self.enthalpy(rho0)
        cs2 = self.gamma * self.K * rho0**(self.gamma_minus_1) / h
        return np.clip(cs2, 0.0, 1.0)

    # Compatibility methods for interface consistency with IdealGasEOS
    def enthalpy_from_rho_p(self, rho0, pressure):
        """Alias for enthalpy - ignores pressure since barotropic."""
        return self.enthalpy(rho0)

    def pressure_from_rho_eps(self, rho0, eps):
        """P(ρ₀) - ignores eps since barotropic."""
        return self.pressure(rho0)


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
    elif eos_type == "tabulated":
        return TabulatedEOS(kwargs.get("table_file"))
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")


# Common presets used in tests and examples
COMMON_EOS = {
    "air": IdealGasEOS(gamma=1.4),            # Diatomic gas
    "monatomic": IdealGasEOS(gamma=5.0 / 3.0),
    "radiation": IdealGasEOS(gamma=4.0 / 3.0),
    "stiff": IdealGasEOS(gamma=2.0),          # Causal limit for ideal gas
    "ns_polytrope": PolytropicEOS(K=100.0, gamma=2.0),
}


def get_common_eos(name: str):
    """Retrieve a predefined EOS configuration by name."""
    if name not in COMMON_EOS:
        raise ValueError(f"Unknown EOS name '{name}'. Available: {list(COMMON_EOS.keys())}")
    return COMMON_EOS[name]
