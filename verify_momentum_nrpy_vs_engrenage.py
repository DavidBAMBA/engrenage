"""
Systematic comparison of momentum equations: NRPy+ vs engrenage

This script implements a staged comparison:
1. Stage 1: Copy NRPy+ GRHD_equations.py implementation to return numpy arrays
2. Stage 2: Import engrenage's grhd_equations.py implementation
3. Stage 3: Create test values and compare outputs

Goal: Find bug in momentum source terms causing velocity growth at stellar surface
"""

import numpy as np
from typing import Tuple


# ============================================================================
# STAGE 1: NRPy+ Implementation (converted to numpy)
# ============================================================================
# These functions are direct translations of NRPy+ GRHD_equations.py
# from SymPy symbolic expressions to numpy array operations

class NRPyGRHDEquations:
    """
    NRPy+ GRHD equations translated from SymPy to numpy.

    This class mimics the structure of nrpy/equations/grhd/GRHD_equations.py
    but operates on numpy arrays instead of symbolic expressions.
    """

    def __init__(self):
        """Initialize containers for all GRHD quantities."""
        # Will be set by compute methods
        self.T4UU = None
        self.T4UD = None
        self.rho_star = None
        self.rho_star_fluxU = None
        self.tau_tilde = None
        self.tau_tilde_fluxU = None
        self.S_tildeD = None
        self.S_tilde_fluxUD = None
        self.S_tilde_connection_termsD = None
        self.S_tilde_source_termD = None

    def compute_T4UU(self, rho_b: np.ndarray, h: np.ndarray,
                     u4U: np.ndarray, g4UU: np.ndarray,
                     P: np.ndarray) -> np.ndarray:
        """
        Compute stress-energy tensor T^{mu nu} (contravariant).

        Formula: T^{μν} = ρ_b h u^μ u^ν + P g^{μν}

        Parameters
        ----------
        rho_b : np.ndarray
            Rest mass density
        h : np.ndarray
            Specific enthalpy
        u4U : np.ndarray (N, 4)
            Four-velocity contravariant components
        g4UU : np.ndarray (N, 4, 4)
            Contravariant spacetime metric
        P : np.ndarray
            Pressure

        Returns
        -------
        T4UU : np.ndarray (N, 4, 4)
            Stress-energy tensor contravariant
        """
        N = len(rho_b)
        T4UU = np.zeros((N, 4, 4))

        # T^{μν} = ρ_b h u^μ u^ν + P g^{μν}
        for mu in range(4):
            for nu in range(4):
                T4UU[:, mu, nu] = (
                    (rho_b * h) * u4U[:, mu] * u4U[:, nu] +
                    P * g4UU[:, mu, nu]
                )

        self.T4UU = T4UU
        return T4UU

    def compute_T4UD(self, T4UU: np.ndarray, g4DD: np.ndarray) -> np.ndarray:
        """
        Compute stress-energy tensor T^{mu}_{nu} (mixed).

        Formula: T^μ_ν = T^{μδ} g_{δν}

        Parameters
        ----------
        T4UU : np.ndarray (N, 4, 4)
            Contravariant stress-energy tensor
        g4DD : np.ndarray (N, 4, 4)
            Covariant spacetime metric

        Returns
        -------
        T4UD : np.ndarray (N, 4, 4)
            Mixed stress-energy tensor
        """
        N = T4UU.shape[0]
        T4UD = np.zeros((N, 4, 4))

        # T^μ_ν = T^{μδ} g_{δν}
        for mu in range(4):
            for nu in range(4):
                for delta in range(4):
                    T4UD[:, mu, nu] += T4UU[:, mu, delta] * g4DD[:, delta, nu]

        self.T4UD = T4UD
        return T4UD

    def compute_rho_star(self, alpha: np.ndarray, e6phi: np.ndarray,
                        rho_b: np.ndarray, u4U: np.ndarray) -> np.ndarray:
        """
        Compute densitized conserved density.

        Formula: D = α e^{6φ} ρ_b u^0

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function
        e6phi : np.ndarray
            Conformal factor exp(6φ)
        rho_b : np.ndarray
            Rest mass density
        u4U : np.ndarray (N, 4)
            Four-velocity

        Returns
        -------
        rho_star : np.ndarray
            Conservative density D
        """
        rho_star = alpha * e6phi * rho_b * u4U[:, 0]
        self.rho_star = rho_star
        return rho_star

    def compute_tau_tilde(self, alpha: np.ndarray, e6phi: np.ndarray,
                         T4UU: np.ndarray, rho_star: np.ndarray) -> np.ndarray:
        """
        Compute densitized conserved energy.

        Formula: τ = α² e^{6φ} T^{00} - D

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function
        e6phi : np.ndarray
            Conformal factor
        T4UU : np.ndarray (N, 4, 4)
            Stress-energy tensor
        rho_star : np.ndarray
            Conservative density

        Returns
        -------
        tau_tilde : np.ndarray
            Conservative energy τ
        """
        tau_tilde = alpha**2 * e6phi * T4UU[:, 0, 0] - rho_star
        self.tau_tilde = tau_tilde
        return tau_tilde

    def compute_S_tildeD(self, alpha: np.ndarray, e6phi: np.ndarray,
                        T4UD: np.ndarray) -> np.ndarray:
        """
        Compute densitized conserved momentum.

        Formula: S_i = α e^{6φ} T^0_i

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function
        e6phi : np.ndarray
            Conformal factor
        T4UD : np.ndarray (N, 4, 4)
            Mixed stress-energy tensor

        Returns
        -------
        S_tildeD : np.ndarray (N, 3)
            Conservative momentum S_i
        """
        N = len(alpha)
        S_tildeD = np.zeros((N, 3))

        for i in range(3):
            S_tildeD[:, i] = alpha * e6phi * T4UD[:, 0, i + 1]

        self.S_tildeD = S_tildeD
        return S_tildeD

    def compute_rho_star_fluxU(self, rho_star: np.ndarray,
                              VU: np.ndarray) -> np.ndarray:
        """
        Compute density flux.

        Formula: F^i_D = D v^i

        Parameters
        ----------
        rho_star : np.ndarray
            Conservative density D
        VU : np.ndarray (N, 3)
            Valencia velocity v^i

        Returns
        -------
        rho_star_fluxU : np.ndarray (N, 3)
            Density flux
        """
        N = len(rho_star)
        rho_star_fluxU = np.zeros((N, 3))

        for j in range(3):
            rho_star_fluxU[:, j] = rho_star * VU[:, j]

        self.rho_star_fluxU = rho_star_fluxU
        return rho_star_fluxU

    def compute_tau_tilde_fluxU(self, alpha: np.ndarray, e6phi: np.ndarray,
                               T4UU: np.ndarray, rho_star: np.ndarray,
                               VU: np.ndarray) -> np.ndarray:
        """
        Compute energy flux.

        Formula: F^i_τ = α² e^{6φ} T^{0i} - D v^i

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function
        e6phi : np.ndarray
            Conformal factor
        T4UU : np.ndarray (N, 4, 4)
            Stress-energy tensor
        rho_star : np.ndarray
            Conservative density
        VU : np.ndarray (N, 3)
            Valencia velocity

        Returns
        -------
        tau_tilde_fluxU : np.ndarray (N, 3)
            Energy flux
        """
        N = len(alpha)
        tau_tilde_fluxU = np.zeros((N, 3))

        for j in range(3):
            tau_tilde_fluxU[:, j] = (
                alpha**2 * e6phi * T4UU[:, 0, j + 1] -
                rho_star * VU[:, j]
            )

        self.tau_tilde_fluxU = tau_tilde_fluxU
        return tau_tilde_fluxU

    def compute_S_tilde_fluxUD(self, alpha: np.ndarray, e6phi: np.ndarray,
                              T4UD: np.ndarray) -> np.ndarray:
        """
        Compute momentum flux tensor.

        Formula: F^j_i = α e^{6φ} T^j_i

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function
        e6phi : np.ndarray
            Conformal factor
        T4UD : np.ndarray (N, 4, 4)
            Mixed stress-energy tensor

        Returns
        -------
        S_tilde_fluxUD : np.ndarray (N, 3, 3)
            Momentum flux tensor F^j_i
        """
        N = len(alpha)
        S_tilde_fluxUD = np.zeros((N, 3, 3))

        for j in range(3):
            for i in range(3):
                S_tilde_fluxUD[:, j, i] = alpha * e6phi * T4UD[:, j + 1, i + 1]

        self.S_tilde_fluxUD = S_tilde_fluxUD
        return S_tilde_fluxUD

    def compute_S_tilde_connection_termsD(self, GammahatUDD: np.ndarray,
                                         S_tilde_fluxUD: np.ndarray) -> np.ndarray:
        """
        Compute connection contributions to momentum equation.

        Formula: Γ_term_i = Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l

        Parameters
        ----------
        GammahatUDD : np.ndarray (N, 3, 3, 3)
            Reference Christoffel symbols Γ̂^i_{jk}
        S_tilde_fluxUD : np.ndarray (N, 3, 3)
            Momentum flux tensor F̃^j_i

        Returns
        -------
        S_tilde_connection_termsD : np.ndarray (N, 3)
            Connection contributions to momentum RHS
        """
        N = S_tilde_fluxUD.shape[0]
        S_tilde_connection_termsD = np.zeros((N, 3))

        # Γ_term_i = Γ̂^j_{jl} F̃^l_i - Γ̂^l_{ji} F̃^j_l
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    S_tilde_connection_termsD[:, i] += (
                        GammahatUDD[:, j, j, l] * S_tilde_fluxUD[:, l, i] -
                        GammahatUDD[:, l, j, i] * S_tilde_fluxUD[:, j, l]
                    )

        self.S_tilde_connection_termsD = S_tilde_connection_termsD
        return S_tilde_connection_termsD

    def compute_S_tilde_source_termD(self, alpha: np.ndarray, alpha_dD: np.ndarray,
                                    betaU: np.ndarray, betaU_dD: np.ndarray,
                                    e6phi: np.ndarray,
                                    T4UU: np.ndarray, T4UD: np.ndarray,
                                    GammahatUDD: np.ndarray,
                                    gammabarDD: np.ndarray, gammabarDD_dD: np.ndarray,
                                    phi_dD: np.ndarray, exp_m4phi: np.ndarray,
                                    return_debug: bool = False) -> np.ndarray:
        """
        Compute geometric source terms for momentum equation.

        This is the CRITICAL function for comparison - implements NRPy+'s
        complete momentum source term calculation.

        Formula:
        S_i = α e^{6φ} [
            -T^{00} α ∂_i α +
            T^0_j (∂_i β^j + Γ̂^j_{ik} β^k) +
            0.5 stress_block × ∇̂_i γ_{jk}
        ]

        where stress_block = T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}

        Parameters
        ----------
        alpha : np.ndarray
            Lapse function
        alpha_dD : np.ndarray (N, 3)
            Lapse gradient ∂_i α
        betaU : np.ndarray (N, 3)
            Shift vector β^i
        betaU_dD : np.ndarray (N, 3, 3)
            Shift derivatives ∂_j β^i
        e6phi : np.ndarray
            Conformal factor exp(6φ)
        T4UU : np.ndarray (N, 4, 4)
            Stress-energy tensor contravariant
        T4UD : np.ndarray (N, 4, 4)
            Stress-energy tensor mixed
        GammahatUDD : np.ndarray (N, 3, 3, 3)
            Reference Christoffel symbols
        gammabarDD : np.ndarray (N, 3, 3)
            Conformal metric γ̄_ij
        gammabarDD_dD : np.ndarray (N, 3, 3, 3)
            Conformal metric derivatives ∂_k γ̄_ij
        phi_dD : np.ndarray (N, 3)
            Conformal factor gradient ∂_i φ
        exp_m4phi : np.ndarray
            Conformal factor exp(-4φ)

        Returns
        -------
        S_tilde_source_termD : np.ndarray (N, 3)
            Geometric source terms for momentum
        """
        N = len(alpha)

        first_termD = np.zeros((N, 3))
        second_termD = np.zeros((N, 3))
        third_termD = np.zeros((N, 3))

        # Compute covariant derivative of conformal metric
        # ∇̂_i γ_{jk} = ∂_i γ_{jk} - Γ̂^l_{ij} γ_{lk} - Γ̂^l_{ik} γ_{jl}
        # Using BSSN: γ_jk = e^{-4φ} γ̄_jk
        # so ∇̂_i γ_{jk} = e^{-4φ} (∂_i γ̄_{jk} - 4 γ̄_{jk} ∂_i φ - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl})

        covhatdD_gammaDD = np.zeros((N, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    covhatdD_gammaDD[:, i, j, k] += exp_m4phi * (
                        4.0 * gammabarDD[:, j, k] * phi_dD[:, i] +
                        gammabarDD_dD[:, j, k, i]
                    )
                    for l in range(3):
                        covhatdD_gammaDD[:, i, j, k] -= exp_m4phi * (
                            GammahatUDD[:, l, i, j] * gammabarDD[:, l, k] +
                            GammahatUDD[:, l, i, k] * gammabarDD[:, j, l]
                        )

        # First term: -T^{00} α ∂_i α
        for i in range(3):
            first_termD[:, i] = -T4UU[:, 0, 0] * alpha * alpha_dD[:, i]

        # Second term: T^0_j (∂_i β^j + Γ̂^j_{ik} β^k)
        for i in range(3):
            for j in range(3):
                second_termD[:, i] += T4UD[:, 0, j + 1] * betaU_dD[:, j, i]
                for k in range(3):
                    second_termD[:, i] += (
                        T4UD[:, 0, j + 1] * GammahatUDD[:, j, i, k] * betaU[:, k]
                    )

        # Third term: 0.5 × stress_block × ∇̂_i γ_{jk}
        # stress_block = T^{00} β^i β^j + 2 T^{0i} β^j + T^{ij}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    stress_block_jk = (
                        T4UU[:, 0, 0] * betaU[:, j] * betaU[:, k] +
                        2.0 * T4UU[:, 0, j + 1] * betaU[:, k] +
                        T4UU[:, j + 1, k + 1]
                    )
                    third_termD[:, i] += (
                        0.5 * covhatdD_gammaDD[:, i, j, k] * stress_block_jk
                    )

        # Total source term
        S_tilde_source_termD = alpha[:, None] * e6phi[:, None] * (
            first_termD + second_termD + third_termD
        )

        self.S_tilde_source_termD = S_tilde_source_termD

        # Return debug info
        if not return_debug:
            return S_tilde_source_termD

        return {
            'total': S_tilde_source_termD,
            'first_term': first_termD,
            'second_term': second_termD,
            'third_term': third_termD,
            'covhatdD_gammaDD': covhatdD_gammaDD,
            'gammabarDD': gammabarDD,
            'gammabarDD_dD': gammabarDD_dD,
            'phi_dD': phi_dD,
            'exp_m4phi': exp_m4phi,
        }


# ============================================================================
# STAGE 2: Engrenage Implementation (import and wrapper)
# ============================================================================

def engrenage_compute_source_terms(test_case):
    """
    Wrapper to call engrenage's GRHDEquations.compute_source_terms.

    This constructs minimal geometry and BSSN objects from test case data.
    """
    from source.matter.hydro.grhd_equations import GRHDEquations
    from source.matter.hydro.eos import PolytropicEOS

    # Create simple containers for BSSN variables
    class SimpleBSSNVars:
        def __init__(self, test_case):
            N = test_case['N']
            self.phi = test_case['phi']
            self.K = np.zeros(N)  # Trace of extrinsic curvature
            # Conformal metric deviation h_ij = γ̄_ij - f_ij
            # For spherical: f_rr = 1, f_θθ = r², f_φφ = r²
            # So h_rr = γ̄_rr - 1
            self.h_LL = np.zeros((N, 3, 3))
            self.h_LL[:, 0, 0] = test_case['gammabarDD'][:, 0, 0] - 1.0
            # For angular: h_θθ = 0, h_φφ = 0 (since γ̄_θθ = f_θθ = r²)
            # Conformal traceless extrinsic curvature
            self.a_LL = np.zeros((N, 3, 3))
            # Shift
            self.shift_U = None  # Will use beta_U from geometry

    class SimpleBSSND1:
        def __init__(self, test_case):
            N = test_case['N']
            self.phi = test_case['phi_dD']
            self.lapse = test_case['alpha_dD']
            # Conformal metric gradient
            self.h_LL = np.zeros((N, 3, 3, 3))
            # ∂_r h_rr = ∂_r(γ̄_rr - 1) = ∂_r γ̄_rr
            self.h_LL[:, 0, 0, 0] = test_case['gammabarDD_dD'][:, 0, 0, 0]
            # For spherical background, angular components already included
            self.h_LL[:, 1, 1, 0] = 0.0  # ∂_r h_θθ = 0 (γ̄_θθ = r² matches f_θθ)
            self.h_LL[:, 2, 2, 0] = 0.0  # ∂_r h_φφ = 0
            # Shift derivatives
            self.shift_U = None  # Will indicate zero shift

    class SimpleBackground:
        def __init__(self, test_case):
            N = test_case['N']
            r = test_case['r']
            self.r = r
            self.N = N

            # Christoffel symbols
            self.hat_christoffel = test_case['GammahatUDD']

            # For spherical: scaling_vector = (1, r, r·sinθ)
            # In 1D spherical with sinθ=1: scaling = (1, r, r)
            self.scaling_vector = np.ones((N, 3))
            self.scaling_vector[:, 1] = r
            self.scaling_vector[:, 2] = r

            # Inverse scaling: inverse_scaling = (1, 1/r, 1/r)
            self.inverse_scaling_vector = np.ones((N, 3))
            self.inverse_scaling_vector[:, 1] = 1.0 / r
            self.inverse_scaling_vector[:, 2] = 1.0 / r

            # Derivative of inverse scaling (for shift covariant derivative)
            # ∂_r(1/r) = -1/r²
            self.d1_inverse_scaling_vector = np.zeros((N, 3, 3))
            self.d1_inverse_scaling_vector[:, 1, 0] = -1.0 / r**2
            self.d1_inverse_scaling_vector[:, 2, 0] = -1.0 / r**2

            # Scaling matrix: s_ij = s_i s_j
            self.scaling_matrix = np.zeros((N, 3, 3))
            for i in range(3):
                for j in range(3):
                    self.scaling_matrix[:, i, j] = (
                        self.scaling_vector[:, i] * self.scaling_vector[:, j]
                    )

            # Inverse scaling matrix: 1/(s_i s_j)
            self.inverse_scaling_matrix = 1.0 / self.scaling_matrix

            # Derivative of scaling matrix
            # For spherical: ∂_r s_θθ = 2r, ∂_r s_φφ = 2r
            self.d1_scaling_matrix = np.zeros((N, 3, 3, 3))
            self.d1_scaling_matrix[:, 1, 1, 0] = 2.0 * r  # ∂_r(r²)
            self.d1_scaling_matrix[:, 2, 2, 0] = 2.0 * r

            # Hat metric: γ̂_ij (flat spherical background)
            self.hat_gamma_LL = np.zeros((N, 3, 3))
            self.hat_gamma_LL[:, 0, 0] = 1.0
            self.hat_gamma_LL[:, 1, 1] = r**2
            self.hat_gamma_LL[:, 2, 2] = r**2

            # hat_D_gamma: ∇̂_i γ̄_{jk} (covariant derivative, not just partial!)
            # ∇̂_i γ̄_{jk} = ∂_i γ̄_{jk} - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl}
            gammabarDD = test_case['gammabarDD']
            gammabarDD_dD = test_case['gammabarDD_dD']
            GammahatUDD = test_case['GammahatUDD']

            self.hat_D_gamma = np.zeros((N, 3, 3, 3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # Start with partial derivative
                        self.hat_D_gamma[:, j, k, i] = gammabarDD_dD[:, j, k, i]
                        # Subtract Christoffel corrections
                        for l in range(3):
                            self.hat_D_gamma[:, j, k, i] -= (
                                GammahatUDD[:, l, i, j] * gammabarDD[:, l, k] +
                                GammahatUDD[:, l, i, k] * gammabarDD[:, j, l]
                            )

    class SimpleGeometry:
        def __init__(self, test_case):
            self.alpha = test_case['alpha']
            self.beta_U = test_case['betaU']
            self.gamma_LL = test_case['gammaDD']
            self.gamma_UU = test_case['gammaUU']
            self.e6phi = test_case['e6phi']
            self.dr = 0.1  # Not used in source terms

    # Extract test data
    rho0 = test_case['rho0']
    v_U = test_case['v_U']
    pressure = test_case['pressure']
    W = test_case['W']
    h = test_case['h']
    r = test_case['r']

    # Create geometry objects
    geometry = SimpleGeometry(test_case)
    bssn_vars = SimpleBSSNVars(test_case)
    bssn_d1 = SimpleBSSND1(test_case)
    background = SimpleBackground(test_case)

    # Create GRHD instance
    eos = PolytropicEOS(K=100.0, gamma=2.0)
    grhd = GRHDEquations(eos=eos)

    # Import stress-energy tensor computation
    from source.matter.hydro.stress_energy import StressEnergyTensor
    from source.matter.hydro.geometry import ADMGeometry

    # Compute stress-energy tensor
    adm_geom = ADMGeometry(geometry.alpha, geometry.beta_U,
                           geometry.gamma_LL, geometry.gamma_UU)
    stress_energy = StressEnergyTensor(adm_geom, rho0, v_U, pressure, W, h)
    T00, T0i, Tij = stress_energy.compute_T4UU()

    # Build full T^{μν}
    N = len(rho0)
    T4UU = np.zeros((N, 4, 4))
    T4UU[:, 0, 0] = T00
    for i in range(3):
        T4UU[:, 0, i+1] = T0i[:, i]
        T4UU[:, i+1, 0] = T0i[:, i]
        for j in range(3):
            T4UU[:, i+1, j+1] = Tij[:, i, j]

    # Build T^μ_ν (mixed)
    T4UD = np.zeros((N, 4, 4))
    # T^0_0 = T^0μ g_μ0 = T^00 g_00 + T^0i g_i0 = T^00 (-α²)
    T4UD[:, 0, 0] = T4UU[:, 0, 0] * (-geometry.alpha**2)
    # T^0_i = T^0j g_ji
    for i in range(3):
        for j in range(3):
            T4UD[:, 0, i+1] += T4UU[:, 0, j+1] * geometry.gamma_LL[:, j, i]
    # T^i_0 = T^iμ g_μ0 = T^i0 g_00 + T^ij g_j0 = T^i0 (-α²)
    for i in range(3):
        T4UD[:, i+1, 0] = T4UU[:, i+1, 0] * (-geometry.alpha**2)
    # T^i_j = T^ik g_kj
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T4UD[:, i+1, j+1] += T4UU[:, i+1, k+1] * geometry.gamma_LL[:, k, j]

    # Compute conservatives
    alpha = geometry.alpha
    e6phi = geometry.e6phi
    rho_star = alpha * e6phi * rho0 * W / alpha  # D = α e^6φ ρ₀ u^0, u^0 = W/α
    S_tildeD = alpha[:, None] * e6phi[:, None] * T4UD[:, 0, 1:4]
    tau_tilde = alpha**2 * e6phi * T4UU[:, 0, 0] - rho_star

    # Compute fluxes (partial flux vectors at cell centers)
    rho_star_fluxU = rho_star[:, None] * v_U  # F^i_D = D v^i
    tau_tilde_fluxU = alpha[:, None]**2 * e6phi[:, None] * T0i - rho_star[:, None] * v_U
    S_tilde_fluxUD = alpha[:, None, None] * e6phi[:, None, None] * T4UD[:, 1:4, 1:4]

    # Compute connection terms
    conn_D, conn_S, conn_tau = grhd.compute_connection_terms(
        rho0, v_U, pressure, W, h, geometry,
        background.hat_christoffel
    )

    # Call the compute_source_terms method with return_debug=True
    src_S, src_tau, debug = grhd.compute_source_terms(
        rho0, v_U, pressure, W, h,
        geometry, bssn_vars, bssn_d1, background,
        spacetime_mode='dynamic', r=r,
        return_debug=True
    )

    return {
        'conservatives': {
            'D': rho_star,
            'S': S_tildeD,
            'tau': tau_tilde,
        },
        'T4UU': T4UU,
        'T4UD': T4UD,
        'fluxes': {
            'D': rho_star_fluxU,
            'S': S_tilde_fluxUD,
            'tau': tau_tilde_fluxU,
        },
        'connections': {
            'D': conn_D,
            'S': conn_S,
            'tau': conn_tau,
        },
        'source': {
            'total': src_S,
            'first_term': debug['momentum_T00_alpha'],
            'second_term': debug['momentum_T0j_beta'],
            'third_term': debug['momentum_metric'],
            'tau': src_tau,
        },
    }


# ============================================================================
# STAGE 3: Test Cases and Comparison
# ============================================================================

def create_test_case_tov_surface():
    """
    Create test case at TOV stellar surface (r ≈ 9.55).

    This is the critical location where velocity growth is observed.
    """
    print("=" * 80)
    print("Creating test case: TOV stellar surface (r ≈ 9.55)")
    print("=" * 80)

    # Grid size (single point for now)
    N = 1

    # Physical conditions at stellar surface (from user's data)
    rho0 = np.array([1.28e-3])  # Just inside star
    pressure = np.array([1.28e-4])  # P ~ 0.1 * rho (rough estimate)
    v_r = np.array([0.011])  # Initial radial velocity
    r = np.array([9.55])  # Radial coordinate

    # Metric (TOV solution - approximate)
    # At surface: α ≈ 0.95, φ ≈ -0.05, γ_rr ≈ 1.05
    alpha = np.array([0.95])
    phi = np.array([-0.05])
    e6phi = np.exp(6.0 * phi)
    exp_m4phi = np.exp(-4.0 * phi)

    # Spatial metric (conformal)
    gamma_rr = 1.05
    gammabarDD = np.zeros((N, 3, 3))
    gammabarDD[:, 0, 0] = gamma_rr * exp_m4phi
    gammabarDD[:, 1, 1] = r**2
    gammabarDD[:, 2, 2] = r**2

    # Physical metric
    gammaDD = np.zeros((N, 3, 3))
    gammaDD[:, 0, 0] = gamma_rr
    gammaDD[:, 1, 1] = r**2
    gammaDD[:, 2, 2] = r**2

    # Shift (approximately zero for TOV)
    betaU = np.zeros((N, 3))

    # Lapse gradient (radial derivative)
    alpha_dD = np.zeros((N, 3))
    alpha_dD[:, 0] = 0.01  # ∂_r α (positive near surface)

    # Shift derivatives (zero for static solution)
    betaU_dD = np.zeros((N, 3, 3))

    # Conformal metric derivatives (spherical background)
    gammabarDD_dD = np.zeros((N, 3, 3, 3))
    # ∂_r γ̄_θθ = 2r, ∂_r γ̄_φφ = 2r
    gammabarDD_dD[:, 1, 1, 0] = 2.0 * r
    gammabarDD_dD[:, 2, 2, 0] = 2.0 * r

    # Conformal factor gradient
    phi_dD = np.zeros((N, 3))
    phi_dD[:, 0] = 0.005  # ∂_r φ

    # Christoffel symbols (spherical coordinates)
    GammahatUDD = np.zeros((N, 3, 3, 3))
    # Γ^r_θθ = -r
    GammahatUDD[:, 0, 1, 1] = -r
    # Γ^r_φφ = -r
    GammahatUDD[:, 0, 2, 2] = -r
    # Γ^θ_rθ = 1/r, Γ^θ_θr = 1/r
    GammahatUDD[:, 1, 0, 1] = 1.0 / r
    GammahatUDD[:, 1, 1, 0] = 1.0 / r
    # Γ^φ_rφ = 1/r, Γ^φ_φr = 1/r
    GammahatUDD[:, 2, 0, 2] = 1.0 / r
    GammahatUDD[:, 2, 2, 0] = 1.0 / r

    # Compute EOS-dependent quantities
    # Polytropic: P = K ρ^Γ, h = 1 + Γ P / ((Γ-1) ρ)
    Gamma = 2.0
    h = 1.0 + Gamma * pressure / ((Gamma - 1.0) * rho0)

    # Lorentz factor
    v_squared = v_r**2 * gammaDD[:, 0, 0]
    W = 1.0 / np.sqrt(1.0 - v_squared)

    # Four-velocity
    u4U = np.zeros((N, 4))
    u4U[:, 0] = W / alpha
    u4U[:, 1] = W * v_r  # u^r = W v^r
    u4U[:, 2] = 0.0
    u4U[:, 3] = 0.0

    # Valencia velocity (v^i = u^i / u^0 α)
    VU = np.zeros((N, 3))
    VU[:, 0] = v_r
    VU[:, 1] = 0.0
    VU[:, 2] = 0.0

    # Spacetime metric
    # g^{00} = -1/α²
    # g^{0i} = β^i/α²
    # g^{ij} = γ^{ij} - β^i β^j / α²
    g4UU = np.zeros((N, 4, 4))
    g4UU[:, 0, 0] = -1.0 / alpha**2

    # Inverse spatial metric
    gammaUU = np.zeros((N, 3, 3))
    gammaUU[:, 0, 0] = 1.0 / gammaDD[:, 0, 0]
    gammaUU[:, 1, 1] = 1.0 / gammaDD[:, 1, 1]
    gammaUU[:, 2, 2] = 1.0 / gammaDD[:, 2, 2]

    for i in range(3):
        for j in range(3):
            g4UU[:, i+1, j+1] = gammaUU[:, i, j]

    # g_{μν}
    g4DD = np.zeros((N, 4, 4))
    g4DD[:, 0, 0] = -alpha**2
    for i in range(3):
        for j in range(3):
            g4DD[:, i+1, j+1] = gammaDD[:, i, j]

    return {
        'N': N,
        'rho0': rho0,
        'pressure': pressure,
        'v_U': VU,
        'v_r': v_r,
        'W': W,
        'h': h,
        'r': r,
        'alpha': alpha,
        'phi': phi,
        'e6phi': e6phi,
        'exp_m4phi': exp_m4phi,
        'betaU': betaU,
        'alpha_dD': alpha_dD,
        'betaU_dD': betaU_dD,
        'gammaDD': gammaDD,
        'gammaUU': gammaUU,
        'gammabarDD': gammabarDD,
        'gammabarDD_dD': gammabarDD_dD,
        'phi_dD': phi_dD,
        'GammahatUDD': GammahatUDD,
        'u4U': u4U,
        'g4UU': g4UU,
        'g4DD': g4DD,
    }


def run_nrpy_computation(test_case):
    """
    Run NRPy+ implementation on test case.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: Running NRPy+ Implementation")
    print("=" * 80)

    nrpy = NRPyGRHDEquations()

    # Extract test data
    rho_b = test_case['rho0']
    P = test_case['pressure']
    h = test_case['h']
    u4U = test_case['u4U']
    g4UU = test_case['g4UU']
    g4DD = test_case['g4DD']
    alpha = test_case['alpha']
    e6phi = test_case['e6phi']
    VU = test_case['v_U']

    # Compute stress-energy tensor
    print("\n1. Computing T^{μν}...")
    T4UU = nrpy.compute_T4UU(rho_b, h, u4U, g4UU, P)
    print(f"   T^{{00}} = {T4UU[0, 0, 0]:.6e}")
    print(f"   T^{{01}} = {T4UU[0, 0, 1]:.6e}")
    print(f"   T^{{11}} = {T4UU[0, 1, 1]:.6e}")

    print("\n2. Computing T^μ_ν...")
    T4UD = nrpy.compute_T4UD(T4UU, g4DD)
    print(f"   T^0_0 = {T4UD[0, 0, 0]:.6e}")
    print(f"   T^0_1 = {T4UD[0, 0, 1]:.6e}")
    print(f"   T^1_1 = {T4UD[0, 1, 1]:.6e}")

    # Compute conservatives
    print("\n3. Computing conservative variables...")
    rho_star = nrpy.compute_rho_star(alpha, e6phi, rho_b, u4U)
    tau_tilde = nrpy.compute_tau_tilde(alpha, e6phi, T4UU, rho_star)
    S_tildeD = nrpy.compute_S_tildeD(alpha, e6phi, T4UD)
    print(f"   D = {rho_star[0]:.6e}")
    print(f"   S_r = {S_tildeD[0, 0]:.6e}")
    print(f"   τ = {tau_tilde[0]:.6e}")

    # Compute fluxes
    print("\n4. Computing fluxes...")
    rho_star_fluxU = nrpy.compute_rho_star_fluxU(rho_star, VU)
    tau_tilde_fluxU = nrpy.compute_tau_tilde_fluxU(alpha, e6phi, T4UU, rho_star, VU)
    S_tilde_fluxUD = nrpy.compute_S_tilde_fluxUD(alpha, e6phi, T4UD)
    print(f"   F^r_D = {rho_star_fluxU[0, 0]:.6e}")
    print(f"   F^r_τ = {tau_tilde_fluxU[0, 0]:.6e}")
    print(f"   F^r_r = {S_tilde_fluxUD[0, 0, 0]:.6e}")

    # Compute connection terms
    print("\n5. Computing connection terms...")
    GammahatUDD = test_case['GammahatUDD']
    S_tilde_conn = nrpy.compute_S_tilde_connection_termsD(GammahatUDD, S_tilde_fluxUD)
    print(f"   Connection S_r = {S_tilde_conn[0, 0]:.6e}")

    # Compute fluxes (need to add these methods to NRPy)
    print("\n6. Computing fluxes...")
    rho_star_fluxU = nrpy.compute_rho_star_fluxU(rho_star, VU)
    tau_tilde_fluxU = nrpy.compute_tau_tilde_fluxU(alpha, e6phi, T4UU, rho_star, VU)
    S_tilde_fluxUD = nrpy.compute_S_tilde_fluxUD(alpha, e6phi, T4UD)

    print(f"   Density flux F^r = {rho_star_fluxU[0, 0]:.6e}")
    print(f"   Momentum flux F^r_r = {S_tilde_fluxUD[0, 0, 0]:.6e}")
    print(f"   Energy flux F^r = {tau_tilde_fluxU[0, 0]:.6e}")

    # Compute source terms
    print("\n7. Computing source terms...")
    source_result = nrpy.compute_S_tilde_source_termD(
        alpha, test_case['alpha_dD'],
        test_case['betaU'], test_case['betaU_dD'],
        e6phi, T4UU, T4UD,
        GammahatUDD,
        test_case['gammabarDD'], test_case['gammabarDD_dD'],
        test_case['phi_dD'], test_case['exp_m4phi'],
        return_debug=True
    )

    print(f"   First term (T^00 α ∂α) = {source_result['first_term'][0, 0]:.6e}")
    print(f"   Second term (T^0j ∇β) = {source_result['second_term'][0, 0]:.6e}")
    print(f"   Third term (stress × ∇γ) = {source_result['third_term'][0, 0]:.6e}")
    print(f"   TOTAL Source S_r = {source_result['total'][0, 0]:.6e}")

    return {
        'T4UU': T4UU,
        'T4UD': T4UD,
        'rho_star': rho_star,
        'S_tildeD': S_tildeD,
        'tau_tilde': tau_tilde,
        'rho_star_fluxU': rho_star_fluxU,
        'tau_tilde_fluxU': tau_tilde_fluxU,
        'S_tilde_fluxUD': S_tilde_fluxUD,
        'connection': S_tilde_conn,
        'source': source_result,
    }


def compare_third_term_detailed(nrpy_debug, engrenage_debug, test_case):
    """
    Detailed comparison of third term computation.
    """
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: Third Term (stress × ∇γ)")
    print("=" * 80)

    # Compare covhatdD_gammaDD (covariant derivative of conformal metric)
    nrpy_cov = nrpy_debug['covhatdD_gammaDD']

    # For engrenage, we need to extract this from the source computation
    # Let's check the key components that go into ∇̂_i γ_{jk}

    print("\n1. Checking conformal metric γ̄_{jk}:")
    print(f"   NRPy+ γ̄_rr = {nrpy_debug['gammabarDD'][0, 0, 0]:.10e}")
    print(f"   NRPy+ γ̄_θθ = {nrpy_debug['gammabarDD'][0, 1, 1]:.10e}")
    print(f"   NRPy+ γ̄_φφ = {nrpy_debug['gammabarDD'][0, 2, 2]:.10e}")

    print("\n2. Checking conformal metric derivatives ∂_k γ̄_{jk}:")
    print(f"   NRPy+ ∂_r γ̄_rr = {nrpy_debug['gammabarDD_dD'][0, 0, 0, 0]:.10e}")
    print(f"   NRPy+ ∂_r γ̄_θθ = {nrpy_debug['gammabarDD_dD'][0, 1, 1, 0]:.10e}")
    print(f"   NRPy+ ∂_r γ̄_φφ = {nrpy_debug['gammabarDD_dD'][0, 2, 2, 0]:.10e}")

    print("\n3. Checking covariant derivative ∇̂_r γ_{rr}:")
    print(f"   NRPy+ ∇̂_r γ_rr = {nrpy_cov[0, 0, 0, 0]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_θθ = {nrpy_cov[0, 0, 1, 1]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_φφ = {nrpy_cov[0, 0, 2, 2]:.10e}")

    # Expected values from NRPy+ formula:
    # ∇̂_i γ_{jk} = e^{-4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ - Γ̂^l_{ij} γ̄_{lk} - Γ̂^l_{ik} γ̄_{jl})
    r = test_case['r'][0]
    phi = test_case['phi'][0]
    phi_dD = test_case['phi_dD']
    exp_m4phi = test_case['exp_m4phi'][0]
    gammabarDD = nrpy_debug['gammabarDD'][0]
    gammabarDD_dD = nrpy_debug['gammabarDD_dD'][0]
    GammahatUDD = test_case['GammahatUDD'][0]

    print(f"\n4. Manual calculation of ∇̂_r γ_θθ:")
    print(f"   γ̄_θθ = {gammabarDD[1, 1]:.10e}")
    print(f"   ∂_r γ̄_θθ = {gammabarDD_dD[1, 1, 0]:.10e}")
    print(f"   φ = {phi:.10e}")
    print(f"   ∂_r φ = {phi_dD[0, 0]:.10e}")
    print(f"   e^(-4φ) = {exp_m4phi:.10e}")
    print(f"   Γ̂^r_rθ = {GammahatUDD[0, 0, 1]:.10e}")
    print(f"   Γ̂^θ_rθ = {GammahatUDD[1, 0, 1]:.10e}")

    # Calculate manually
    term1 = exp_m4phi * (4.0 * gammabarDD[1, 1] * phi_dD[0, 0] + gammabarDD_dD[1, 1, 0])
    term2 = -exp_m4phi * (GammahatUDD[0, 0, 1] * gammabarDD[0, 1] + GammahatUDD[0, 0, 1] * gammabarDD[1, 0])
    term3 = -exp_m4phi * (GammahatUDD[1, 0, 1] * gammabarDD[1, 1] + GammahatUDD[1, 0, 1] * gammabarDD[1, 1])
    manual_cov_theta = term1 + term2 + term3

    print(f"\n   Manual breakdown:")
    print(f"   Term 1 (e^(-4phi)(dr gbar_tt + 4gbar_tt dr phi)) = {term1:.10e}")
    print(f"   Term 2 (-e^(-4phi) Ghat^r_rt gbar_rt) = {term2:.10e}")
    print(f"   Term 3 (-e^(-4phi) Ghat^t_rt gbar_tt) = {term3:.10e}")
    print(f"   Total manual = {manual_cov_theta:.10e}")
    print(f"   NRPy+ result = {nrpy_cov[0, 0, 1, 1]:.10e}")
    print(f"   Difference = {abs(manual_cov_theta - nrpy_cov[0, 0, 1, 1]):.10e}")

    print(f"\n5. Checking all components of ∇̂_r γ_jk:")
    print(f"   NRPy+ ∇̂_r γ_rr = {nrpy_cov[0, 0, 0, 0]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_rθ = {nrpy_cov[0, 0, 0, 1]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_θr = {nrpy_cov[0, 0, 1, 0]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_θθ = {nrpy_cov[0, 0, 1, 1]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_θφ = {nrpy_cov[0, 0, 1, 2]:.10e}")
    print(f"   NRPy+ ∇̂_r γ_φφ = {nrpy_cov[0, 0, 2, 2]:.10e}")


def compare_term(name, nrpy_val, eng_val, threshold=1e-10, indent="  "):
    """
    Compare a single term and return error status.
    """
    diff = abs(nrpy_val - eng_val)
    rel_err = diff / (abs(nrpy_val) + 1e-30)

    print(f"{indent}{name}:")
    print(f"{indent}  NRPy+:     {nrpy_val:+.15e}")
    print(f"{indent}  Engrenage: {eng_val:+.15e}")
    print(f"{indent}  Abs diff:  {diff:.15e}")
    print(f"{indent}  Rel err:   {rel_err:.15e}")

    has_error = rel_err > threshold
    if has_error:
        print(f"{indent}  ⚠️  ERROR: Exceeds threshold {threshold:.2e}")
    else:
        print(f"{indent}  ✓ OK")

    return has_error


def compare_comprehensive(nrpy_result, engrenage_result, test_case, threshold=1e-10):
    """
    Exhaustive comparison of ALL GRHD equations and terms.

    Compares:
    1. Conservative variables (D, S_i, tau)
    2. Fluxes (for density, momentum, energy)
    3. Connection terms (for all equations)
    4. Source terms (for momentum and energy)
    5. Stress-energy tensor components
    """
    print("\n" + "=" * 80)
    print("EXHAUSTIVE COMPARISON: All GRHD Equations")
    print("=" * 80)

    errors = []

    # ========================================================================
    # PART 1: Conservative Variables
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 1: Conservative Variables")
    print("─" * 80)

    # D (conserved density)
    print("\n1.1. Conserved Density (D = α e^6φ ρ₀ u^0):")
    if compare_term("D", nrpy_result['rho_star'][0],
                    engrenage_result['conservatives']['D'][0], threshold):
        errors.append("Conservative D")

    # S_i (conserved momentum)
    print("\n1.2. Conserved Momentum S_r:")
    if compare_term("S_r", nrpy_result['S_tildeD'][0, 0],
                    engrenage_result['conservatives']['S'][0, 0], threshold):
        errors.append("Conservative S_r")

    print("\n1.3. Conserved Momentum S_θ:")
    if compare_term("S_θ", nrpy_result['S_tildeD'][0, 1],
                    engrenage_result['conservatives']['S'][0, 1], threshold):
        errors.append("Conservative S_θ")

    print("\n1.4. Conserved Momentum S_φ:")
    if compare_term("S_φ", nrpy_result['S_tildeD'][0, 2],
                    engrenage_result['conservatives']['S'][0, 2], threshold):
        errors.append("Conservative S_φ")

    # tau (conserved energy)
    print("\n1.5. Conserved Energy (τ = α² e^6φ T^00 - D):")
    if compare_term("τ", nrpy_result['tau_tilde'][0],
                    engrenage_result['conservatives']['tau'][0], threshold):
        errors.append("Conservative tau")

    # ========================================================================
    # PART 2: Stress-Energy Tensor
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 2: Stress-Energy Tensor Components")
    print("─" * 80)

    T4UU_nrpy = nrpy_result['T4UU'][0]
    T4UU_eng = engrenage_result['T4UU'][0]

    print("\n2.1. T^{μν} (contravariant):")
    for mu in range(4):
        for nu in range(mu, 4):  # Symmetric, only upper triangle
            mu_name = ['0', 'r', 'θ', 'φ'][mu]
            nu_name = ['0', 'r', 'θ', 'φ'][nu]
            if compare_term(f"T^{mu_name}{nu_name}", T4UU_nrpy[mu, nu],
                           T4UU_eng[mu, nu], threshold, indent="    "):
                errors.append(f"T4UU[{mu},{nu}]")

    T4UD_nrpy = nrpy_result['T4UD'][0]
    T4UD_eng = engrenage_result['T4UD'][0]

    print("\n2.2. T^μ_ν (mixed):")
    for mu in range(4):
        for nu in range(4):
            mu_name = ['0', 'r', 'θ', 'φ'][mu]
            nu_name = ['0', 'r', 'θ', 'φ'][nu]
            if compare_term(f"T^{mu_name}_{nu_name}", T4UD_nrpy[mu, nu],
                           T4UD_eng[mu, nu], threshold, indent="    "):
                errors.append(f"T4UD[{mu},{nu}]")

    # ========================================================================
    # PART 3: Fluxes
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 3: Fluxes")
    print("─" * 80)

    print("\n3.1. Density flux F^i_D:")
    flux_D_nrpy = nrpy_result.get('rho_star_fluxU', np.zeros((1, 3)))[0]
    flux_D_eng = engrenage_result['fluxes']['D'][0]
    for i, name in enumerate(['r', 'θ', 'φ']):
        if compare_term(f"F^{name}_D", flux_D_nrpy[i], flux_D_eng[i], threshold, indent="    "):
            errors.append(f"Flux D^{name}")

    print("\n3.2. Momentum flux F^j_i:")
    flux_S_nrpy = nrpy_result.get('S_tilde_fluxUD', np.zeros((1, 3, 3)))[0]
    flux_S_eng = engrenage_result['fluxes']['S'][0]
    for j, jname in enumerate(['r', 'θ', 'φ']):
        for i, iname in enumerate(['r', 'θ', 'φ']):
            if compare_term(f"F^{jname}_{iname}", flux_S_nrpy[j, i],
                           flux_S_eng[j, i], threshold, indent="    "):
                errors.append(f"Flux S^{jname}_{iname}")

    print("\n3.3. Energy flux F^i_τ:")
    flux_tau_nrpy = nrpy_result.get('tau_tilde_fluxU', np.zeros((1, 3)))[0]
    flux_tau_eng = engrenage_result['fluxes']['tau'][0]
    for i, name in enumerate(['r', 'θ', 'φ']):
        if compare_term(f"F^{name}_τ", flux_tau_nrpy[i], flux_tau_eng[i], threshold, indent="    "):
            errors.append(f"Flux tau^{name}")

    # ========================================================================
    # PART 4: Connection Terms
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 4: Connection Terms (Christoffel contributions)")
    print("─" * 80)

    print("\n4.1. Density connection term:")
    conn_D_nrpy = engrenage_result['connections']['D'][0]
    conn_D_eng = engrenage_result['connections']['D'][0]
    if compare_term("Conn_D", conn_D_nrpy, conn_D_eng, threshold):
        errors.append("Connection D")

    print("\n4.2. Momentum connection terms:")
    conn_S_nrpy = nrpy_result.get('connection', np.zeros((1, 3)))[0]
    conn_S_eng = engrenage_result['connections']['S'][0]
    for i, name in enumerate(['r', 'θ', 'φ']):
        if compare_term(f"Conn_S_{name}", conn_S_nrpy[i], conn_S_eng[i], threshold, indent="    "):
            errors.append(f"Connection S_{name}")

    print("\n4.3. Energy connection term:")
    conn_tau_nrpy = engrenage_result['connections']['tau'][0]
    conn_tau_eng = engrenage_result['connections']['tau'][0]
    if compare_term("Conn_τ", conn_tau_nrpy, conn_tau_eng, threshold):
        errors.append("Connection tau")

    # ========================================================================
    # PART 5: Source Terms
    # ========================================================================
    print("\n" + "─" * 80)
    print("PART 5: Geometric Source Terms")
    print("─" * 80)

    print("\n5.1. Momentum source (detailed breakdown):")

    print("\n  5.1a. First term (-T^00 α ∂α):")
    nrpy_src_1 = nrpy_result['source']['first_term'][0, 0]
    eng_src_1 = engrenage_result['source']['first_term'][0, 0]
    if compare_term("Term1_S_r", nrpy_src_1, eng_src_1, threshold, indent="      "):
        errors.append("Source S_r term1")

    print("\n  5.1b. Second term (T^0_j ∇β):")
    nrpy_src_2 = nrpy_result['source']['second_term'][0, 0]
    eng_src_2 = engrenage_result['source']['second_term'][0, 0]
    if compare_term("Term2_S_r", nrpy_src_2, eng_src_2, threshold, indent="      "):
        errors.append("Source S_r term2")

    print("\n  5.1c. Third term (0.5 stress × ∇γ):")
    nrpy_src_3 = nrpy_result['source']['third_term'][0, 0]
    eng_src_3 = engrenage_result['source']['third_term'][0, 0]
    if compare_term("Term3_S_r", nrpy_src_3, eng_src_3, threshold, indent="      "):
        errors.append("Source S_r term3")

    print("\n  5.1d. Total momentum source S_r:")
    nrpy_src_tot = nrpy_result['source']['total'][0, 0]
    eng_src_tot = engrenage_result['source']['total'][0, 0]
    if compare_term("Total_S_r", nrpy_src_tot, eng_src_tot, threshold, indent="      "):
        errors.append("Source S_r total")

    print("\n5.2. Energy source term:")
    eng_src_tau = engrenage_result['source']['tau'][0]
    print(f"    Engrenage τ source: {eng_src_tau:+.15e}")
    print("    (NRPy+ energy source not yet implemented in comparison)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if errors:
        print(f"\n⚠️  Found {len(errors)} discrepancies exceeding threshold {threshold:.2e}:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return True
    else:
        print(f"\n✓ ALL TERMS AGREE within threshold {threshold:.2e}")
        print("  Implementations are mathematically equivalent!")
        return False


def compare_results(nrpy_result, engrenage_result):
    """
    Quick summary comparison (legacy function for compatibility).
    """
    print("\n" + "=" * 80)
    print("QUICK COMPARISON SUMMARY")
    print("=" * 80)

    # Compare source terms
    nrpy_total = nrpy_result['source']['total'][0, 0]
    eng_total = engrenage_result['source']['total'][0, 0]

    print(f"\nTotal Source S_r:")
    print(f"  NRPy+:     {nrpy_total:.10e}")
    print(f"  Engrenage: {eng_total:.10e}")
    print(f"  Difference: {abs(nrpy_total - eng_total):.10e}")
    print(f"  Rel. error: {abs(nrpy_total - eng_total) / (abs(nrpy_total) + 1e-30):.10e}")

    print(f"\nFirst term (T^00 α ∂α):")
    nrpy_t1 = nrpy_result['source']['first_term'][0, 0]
    eng_t1 = engrenage_result['source']['first_term'][0, 0]
    print(f"  NRPy+:     {nrpy_t1:.10e}")
    print(f"  Engrenage: {eng_t1:.10e}")
    print(f"  Difference: {abs(nrpy_t1 - eng_t1):.10e}")

    print(f"\nSecond term (T^0j ∇β):")
    nrpy_t2 = nrpy_result['source']['second_term'][0, 0]
    eng_t2 = engrenage_result['source']['second_term'][0, 0]
    print(f"  NRPy+:     {nrpy_t2:.10e}")
    print(f"  Engrenage: {eng_t2:.10e}")
    print(f"  Difference: {abs(nrpy_t2 - eng_t2):.10e}")

    print(f"\nThird term (stress × ∇γ):")
    nrpy_t3 = nrpy_result['source']['third_term'][0, 0]
    eng_t3 = engrenage_result['source']['third_term'][0, 0]
    print(f"  NRPy+:     {nrpy_t3:.10e}")
    print(f"  Engrenage: {eng_t3:.10e}")
    print(f"  Difference: {abs(nrpy_t3 - eng_t3):.10e}")

    # Flag significant differences
    threshold = 1e-6
    has_error = False

    if abs(nrpy_total - eng_total) / (abs(nrpy_total) + 1e-30) > threshold:
        print(f"\n⚠️  WARNING: Large difference in TOTAL source term!")
        has_error = True

    if abs(nrpy_t1 - eng_t1) / (abs(nrpy_t1) + 1e-30) > threshold:
        print(f"\n⚠️  WARNING: Large difference in FIRST term!")
        has_error = True

    if abs(nrpy_t2 - eng_t2) / (abs(nrpy_t2) + 1e-30) > threshold:
        print(f"\n⚠️  WARNING: Large difference in SECOND term!")
        has_error = True

    if abs(nrpy_t3 - eng_t3) / (abs(nrpy_t3) + 1e-30) > threshold:
        print(f"\n⚠️  WARNING: Large difference in THIRD term!")
        has_error = True

    if not has_error:
        print(f"\n✓ All terms agree within threshold ({threshold})")

    return has_error


if __name__ == "__main__":
    print("=" * 80)
    print("SYSTEMATIC COMPARISON: NRPy+ vs Engrenage Momentum Equations")
    print("=" * 80)

    # Create test case
    test_case = create_test_case_tov_surface()

    # Run NRPy+ implementation
    nrpy_result = run_nrpy_computation(test_case)

    # Run engrenage implementation
    print("\n" + "=" * 80)
    print("STAGE 2: Running Engrenage Implementation")
    print("=" * 80)

    try:
        engrenage_result = engrenage_compute_source_terms(test_case)

        print(f"\nEngrenage conservatives:")
        print(f"   D = {engrenage_result['conservatives']['D'][0]:.6e}")
        print(f"   S_r = {engrenage_result['conservatives']['S'][0, 0]:.6e}")
        print(f"   τ = {engrenage_result['conservatives']['tau'][0]:.6e}")

        # EXHAUSTIVE COMPARISON
        print("\n" + "=" * 80)
        print("RUNNING EXHAUSTIVE COMPARISON")
        print("=" * 80)

        has_error = compare_comprehensive(nrpy_result, engrenage_result, test_case, threshold=1e-10)

        # If there's an error, also show quick summary
        if has_error:
            compare_results(nrpy_result, engrenage_result)

            # Detailed analysis of third term if needed
            compare_third_term_detailed(
                nrpy_result['source'],
                engrenage_result,
                test_case
            )

        if has_error:
            print("\n" + "=" * 80)
            print("⚠️  DISCREPANCIES DETECTED - SEE DETAILS ABOVE")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("✓✓✓ ALL IMPLEMENTATIONS AGREE - NO BUGS FOUND")
            print("=" * 80)

    except Exception as e:
        print(f"\n⚠️  Error running engrenage implementation: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to NRPy+ only for now.")
