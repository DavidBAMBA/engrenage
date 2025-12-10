"""

This module provides numerical (NumPy-based) implementations of the GRHD equations

Each function is a direct translation from SymPy to NumPy, preserving:
- The exact same function names
- The exact same variable names
- The exact same loop structure
- The exact same computation order

Reference: Terrence Pierre Jacques,  GRHD implementation
Author: Terrence Pierre Jacques
        terrencepierrej **at** gmail **dot* com
"""

import numpy as np
from typing import Dict

SPACEDIM = 3


class GRHD_Equations_NumPy:
    """
    Numerical implementation of GRHD equations 

    This class computes all hydrodynamical quantities using NumPy arrays instead of
    SymPy expressions, but follows the exact same computational structure.

    """

    def __init__(self, N: int):
        """
        Initialize storage for GRHD quantities.

        Args:
            N: Number of spatial points
        """
        self.N = N

        # Geometry (ADM variables, set externally)
        self.alpha = None          # Lapse (N,)
        self.betaU = None          # Shift β^i (N, 3)
        self.betaU_dD = None       # Shift derivatives ∂_j β^i (N, 3, 3)
        self.gammaDD = None        # Physical metric γ_{ij} (N, 3, 3)
        self.gammaUU = None        # Inverse metric γ^{ij} (N, 3, 3)
        self.e6phi = None          # e^{6φ} (N,)
        self.KDD = None            # Extrinsic curvature K_{ij} (N, 3, 3)
        self.alpha_dD = None       # Lapse derivatives ∂_i α (N, 3)
        self.GammahatUDD = None    # Reference Christoffel Γ̂^i_{jk} (N, 3, 3, 3)

        # Primitive hydrodynamical variables (set externally)
        self.rho_b = None          # Rest-mass density ρ_b (N,)
        self.P = None              # Pressure P (N,)
        self.h = None              # Specific enthalpy h (N,)
        self.VU = None             # Valencia 3-velocity v^i (N, 3)
        self.u4U = None            # 4-velocity u^μ (N, 4)

        # Declare class variables that will be defined later
        # (Corresponds to GRHD_equations.py lines 100-134)
        self.T4UU = None           # Stress-energy T^{μν}
        self.T4UD = None           # Mixed stress-energy T^μ_ν

        self.rho_star = None       # Conserved density ρ*
        self.tau_tilde = None      # Conserved energy τ̃
        self.S_tildeD = None       # Conserved momentum S̃_i (N, 3)

        self.rho_star_fluxU = None       # Density flux F^j_ρ (N, 3)
        self.tau_tilde_fluxU = None      # Energy flux F^j_τ (N, 3)
        self.S_tilde_fluxUD = None       # Momentum flux F^j_i (N, 3, 3)

        self.tau_source_term = None              # Energy source S_τ (N,)
        self.rho_star_connection_term = None     # Density connection (N,)
        self.tau_connection_term = None          # Energy connection (N,)
        self.S_tilde_connection_termsD = None    # Momentum connection (N, 3)
        self.S_tilde_source_termD = None         # Momentum source (N, 3)

    def compute_T4UU(self) -> None:
        """
        Define T^{mu nu} (a 4-dimensional tensor).

        Corresponds to GRHD_equations.py lines 136-174
        """
        # GRHD_equations.py line 138-144
        gammaDD = self.gammaDD
        betaU = self.betaU
        alpha = self.alpha
        rho_b = self.rho_b
        P = self.P
        h = self.h
        u4U = self.u4U

        # define g^{mu nu} in terms of the ADM quantities:
        # GRHD_equations.py line 147 (calls ADM_to_g4UU)
        # We compute it explicitly here

        # g^{tt} = -1/α²
        g4UU_00 = -1.0 / (alpha ** 2)

        # g^{ti} = β^i/α²
        g4UU_0i = np.zeros((self.N, 3))
        for i in range(3):
            g4UU_0i[:, i] = betaU[:, i] / (alpha ** 2)

        # g^{ij} = γ^{ij} - β^i β^j / α²
        g4UU_spatial = np.zeros((self.N, 3, 3))
        for i in range(3):
            for j in range(3):
                g4UU_spatial[:, i, j] = self.gammaUU[:, i, j] - betaU[:, i] * betaU[:, j] / (alpha ** 2)

        # compute T^{mu nu}
        # GRHD_equations.py line 156-161
        self.T4UU = {}

        # T^{00}
        self.T4UU['00'] = (rho_b * h) * u4U[:, 0] * u4U[:, 0] + P * g4UU_00

        # T^{0i}
        self.T4UU['0i'] = np.zeros((self.N, 3))
        for i in range(3):
            self.T4UU['0i'][:, i] = (rho_b * h) * u4U[:, 0] * u4U[:, i + 1] + P * g4UU_0i[:, i]

        # T^{ij}
        self.T4UU['ij'] = np.zeros((self.N, 3, 3))
        for i in range(3):
            for j in range(3):
                self.T4UU['ij'][:, i, j] = (rho_b * h) * u4U[:, i + 1] * u4U[:, j + 1] + P * g4UU_spatial[:, i, j]

    def compute_T4UD(self) -> None:
        """
        Define T^{mu}_{nu} (a 4-dimensional tensor).

        Corresponds to GRHD_equations.py lines 176-213
        """
        # GRHD_equations.py line 178-180
        gammaDD = self.gammaDD
        betaU = self.betaU
        alpha = self.alpha

        # compute T^mu_nu = T^{mu delta} g_{delta nu}, needed for S_tilde flux.
        # we'll need g_{alpha nu} in terms of ADM quantities:
        # GRHD_equations.py line 184 (calls ADM_to_g4DD)

        # g_{tt} = -α² + β_k β^k
        beta_lower = np.zeros((self.N, 3))
        for i in range(3):
            for j in range(3):
                beta_lower[:, i] += gammaDD[:, i, j] * betaU[:, j]

        beta_squared = np.zeros(self.N)
        for i in range(3):
            beta_squared += betaU[:, i] * beta_lower[:, i]

        g4DD_00 = -alpha ** 2 + beta_squared

        # g_{ti} = β_i (lowered)
        g4DD_0i = beta_lower

        # g_{ij} = γ_{ij}
        g4DD_spatial = gammaDD

        # GRHD_equations.py line 192-198
        self.T4UD = {}

        # T^0_0 = T^{00} g_{00} + T^{0i} g_{i0}
        self.T4UD['0_0'] = self.T4UU['00'] * g4DD_00
        for i in range(3):
            self.T4UD['0_0'] += self.T4UU['0i'][:, i] * g4DD_0i[:, i]

        # T^0_j = T^{00} g_{0j} + T^{0i} g_{ij}
        self.T4UD['0_j'] = np.zeros((self.N, 3))
        for j in range(3):
            self.T4UD['0_j'][:, j] = self.T4UU['00'] * g4DD_0i[:, j]
            for i in range(3):
                self.T4UD['0_j'][:, j] += self.T4UU['0i'][:, i] * g4DD_spatial[:, i, j]

        # T^i_j = T^{i0} g_{0j} + T^{ik} g_{kj}
        self.T4UD['i_j'] = np.zeros((self.N, 3, 3))
        for i in range(3):
            for j in range(3):
                self.T4UD['i_j'][:, i, j] = self.T4UU['0i'][:, i] * g4DD_0i[:, j]
                for k in range(3):
                    self.T4UD['i_j'][:, i, j] += self.T4UU['ij'][:, i, k] * g4DD_spatial[:, k, j]

    def compute_rho_star(self) -> None:
        """
        Compute densitized conserved density.

        Corresponds to GRHD_equations.py lines 218-225
        """
        # GRHD_equations.py line 220-223
        alpha = self.alpha
        e6phi = self.e6phi
        rho_b = self.rho_b
        u4U = self.u4U

        # Compute rho_star:
        # GRHD_equations.py line 224
        self.rho_star = alpha * e6phi * rho_b * u4U[:, 0]

    def compute_tau_tilde(self) -> None:
        """
        Compute densitized conserved energy.

        Corresponds to GRHD_equations.py lines 236-244
        """
        # GRHD_equations.py line 238-241
        alpha = self.alpha
        e6phi = self.e6phi
        T4UU = self.T4UU
        rho_star = self.rho_star

        # GRHD_equations.py line 243
        self.tau_tilde = alpha**2 * e6phi * T4UU['00'] - rho_star

    def compute_S_tildeD(self) -> None:
        """
        Compute densitized conserved momentum.

        Corresponds to GRHD_equations.py lines 246-256
        """
        # GRHD_equations.py line 248-250
        alpha = self.alpha
        e6phi = self.e6phi
        T4UD = self.T4UD

        # GRHD_equations.py line 249-250
        self.S_tildeD = np.zeros((self.N, 3))
        for i in range(3):
            self.S_tildeD[:, i] = alpha * e6phi * T4UD['0_j'][:, i]

    def compute_rho_star_fluxU(self) -> None:
        """
        Density flux term.

        Corresponds to GRHD_equations.py lines 269-278
        """
        # GRHD_equations.py line 271-272
        VU = self.VU
        rho_star = self.rho_star

        # GRHD_equations.py line 272-275
        self.rho_star_fluxU = np.zeros((self.N, 3))
        for j in range(3):
            self.rho_star_fluxU[:, j] = rho_star * VU[:, j]

    def compute_tau_tilde_fluxU(self) -> None:
        """
        Energy flux term.

        Corresponds to GRHD_equations.py lines 287-299
        """
        # GRHD_equations.py line 289-293
        alpha = self.alpha
        e6phi = self.e6phi
        VU = self.VU
        T4UU = self.T4UU
        rho_star = self.rho_star

        # GRHD_equations.py line 292-295
        self.tau_tilde_fluxU = np.zeros((self.N, 3))
        for j in range(3):
            self.tau_tilde_fluxU[:, j] = (
                alpha**2 * e6phi * T4UU['0i'][:, j] - rho_star * VU[:, j]
            )

    def compute_S_tilde_fluxUD(self) -> None:
        """
        Momentum flux term.

        Corresponds to GRHD_equations.py lines 301-313
        """
        # GRHD_equations.py line 303-304
        alpha = self.alpha
        e6phi = self.e6phi
        T4UD = self.T4UD

        # GRHD_equations.py line 306-309
        self.S_tilde_fluxUD = np.zeros((self.N, 3, 3))
        for j in range(3):
            for i in range(3):
                self.S_tilde_fluxUD[:, j, i] = alpha * e6phi * T4UD['i_j'][:, j, i]

    def compute_tau_source_term(self) -> None:
        """
        Source terms for energy equation.

        Corresponds to GRHD_equations.py lines 315-333
        """
        # GRHD_equations.py line 317-323
        KDD = self.KDD
        betaU = self.betaU
        alpha = self.alpha
        e6phi = self.e6phi
        alpha_dD = self.alpha_dD
        T4UU = self.T4UU

        # GRHD_equations.py line 318
        self.tau_source_term = np.zeros(self.N)
        term1 = np.zeros(self.N)
        term2 = np.zeros(self.N)

        # Term 1:
        # GRHD_equations.py line 319-323
        for i in range(3):
            for j in range(3):
                term1 += (
                    T4UU['00'] * betaU[:, i] * betaU[:, j]
                    + 2 * T4UU['0i'][:, i] * betaU[:, j]
                    + T4UU['ij'][:, i, j]
                ) * KDD[:, i, j]

        # Term 2:
        # GRHD_equations.py line 324-327
        for i in range(3):
            term2 += (
                -(T4UU['00'] * betaU[:, i] + T4UU['0i'][:, i]) * alpha_dD[:, i]
            )

        # --- TEMPORARY DEBUG: Store sub-terms before scaling ---
        self._tau_source_Kij_term = term1.copy()
        self._tau_source_dalpha_term = term2.copy()
        # --- END TEMPORARY DEBUG ---

        # Term 3:
        # GRHD_equations.py line 328-329
        self.tau_source_term = (term1 + term2) * alpha * e6phi

    def compute_rho_star__Ye_star__and_tau_connection_terms(self) -> None:
        """
        Source terms from connection coefficients, for density, electron fraction, and energy equations.

        Corresponds to GRHD_equations.py lines 335-352
        """
        # GRHD_equations.py line 338-339
        self.rho_star_connection_term = np.zeros(self.N)
        self.tau_connection_term = np.zeros(self.N)

        # GRHD_equations.py line 340-351
        for i in range(3):
            for j in range(3):
                self.rho_star_connection_term += (
                    self.GammahatUDD[:, i, i, j] * self.rho_star_fluxU[:, j]
                )
                self.tau_connection_term += (
                    self.GammahatUDD[:, i, i, j] * self.tau_tilde_fluxU[:, j]
                )

    def compute_S_tilde_connection_termsD(self) -> None:
        """
        Source terms from connection coefficients for momentum equation.

        Corresponds to GRHD_equations.py lines 354-365
        """
        # GRHD_equations.py line 357
        self.S_tilde_connection_termsD = np.zeros((self.N, 3))

        # GRHD_equations.py line 358-363
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    self.S_tilde_connection_termsD[:, i] += (
                        self.GammahatUDD[:, j, j, l] * self.S_tilde_fluxUD[:, l, i]
                        - self.GammahatUDD[:, l, j, i] * self.S_tilde_fluxUD[:, j, l]
                    )

    def compute_S_tilde_source_termD(self, gammabarDD, gammabarDD_dD, phi_dD) -> None:
        """
        Source terms for momentum equation.

        Corresponds to GRHD_equations.py lines 367-409

        Args:
            gammabarDD: Conformal metric γ̄_{ij} (N, 3, 3)
            gammabarDD_dD: Derivatives ∂_k γ̄_{ij} (N, 3, 3, 3)
            phi_dD: Conformal factor derivatives ∂_i φ (N, 3)
        """
        # GRHD_equations.py line 369-376
        alpha = self.alpha
        alpha_dD = self.alpha_dD
        betaU = self.betaU
        betaU_dD = self.betaU_dD
        e6phi = self.e6phi
        T4UU = self.T4UU
        T4UD = self.T4UD

        # GRHD_equations.py line 378-380
        first_termD = np.zeros((self.N, 3))
        second_termD = np.zeros((self.N, 3))
        third_termD = np.zeros((self.N, 3))
        self.S_tilde_source_termD = np.zeros((self.N, 3))

        # Covariant derivative of metric ∇̂_i γ_{jk}
        # GRHD_equations.py line 382-383
        # Note: exp_m4phi^{-1} = e^{4φ} = (e^{6φ})^{2/3}
        exp_m4phi_inv = e6phi ** (2.0 / 3.0)

        # GRHD_equations.py line 379
        covhatdD_gammaDD = np.zeros((self.N, 3, 3, 3))

        # GRHD_equations.py line 384-390
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    covhatdD_gammaDD[:, i, j, k] += exp_m4phi_inv * (
                        4.0 * gammabarDD[:, j, k] * phi_dD[:, i]
                        + gammabarDD_dD[:, j, k, i]
                    )
                    for l in range(3):
                        covhatdD_gammaDD[:, i, j, k] -= exp_m4phi_inv * (
                            self.GammahatUDD[:, l, i, j] * gammabarDD[:, l, k]
                            + self.GammahatUDD[:, l, i, k] * gammabarDD[:, j, l]
                        )

        # GRHD_equations.py line 392-406
        for i in range(3):
            # First term
            first_termD[:, i] -= T4UU['00'] * alpha * alpha_dD[:, i]

            for j in range(3):
                # Second term
                second_termD[:, i] += T4UD['0_j'][:, j] * betaU_dD[:, j, i]

                for k in range(3):
                    second_termD[:, i] += (
                        T4UD['0_j'][:, j] * self.GammahatUDD[:, j, i, k] * betaU[:, k]
                    )

                    # Third term
                    third_termD[:, i] += (
                        0.5 * covhatdD_gammaDD[:, i, j, k] * (
                            T4UU['00'] * betaU[:, j] * betaU[:, k]
                            + 2.0 * T4UU['0i'][:, j] * betaU[:, k]
                            + T4UU['ij'][:, j, k]
                        )
                    )

        # --- TEMPORARY DEBUG: Store sub-terms before scaling ---
        self._momentum_source_T00_alpha_term = first_termD.copy()
        self._momentum_source_T0j_beta_term = second_termD.copy()
        self._momentum_source_metric_term = third_termD.copy()
        self._covhatdD_gammaDD_debug = covhatdD_gammaDD.copy()
        # --- END TEMPORARY DEBUG ---

        # GRHD_equations.py line 408-409
        for i in range(3):
            self.S_tilde_source_termD[:, i] += (
                alpha * e6phi * (first_termD[:, i] + second_termD[:, i] + third_termD[:, i])
            )
