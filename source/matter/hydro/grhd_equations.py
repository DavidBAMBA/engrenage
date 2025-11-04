"""
GRHD Equations Module

This module provides the GRHDEquations class that orchestrates the complete
General Relativistic Hydrodynamics evolution equations using the Valencia formulation.

Classes:
    GRHDEquations: Main class for GRHD evolution
"""

import numpy as np
from typing import Tuple, Optional

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_bar_A_LL,
    get_hat_D_bar_gamma_LL,
)
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.geometry import ADMGeometry, ValenciaGeometry
from source.matter.hydro.stress_energy import StressEnergyTensor
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.atmosphere import AtmosphereParams


class GRHDEquations:
    """
    Orchestrates GRHD evolution equations in the Valencia formulation.

    This class implements the conservative evolution equations:
    ∂_t(U) + ∂_j(F̃^j) = S + connection_divergence_terms

    where:
    - U = (D, S_i, τ) are the conservative variables
    - F̃^j are the densitized fluxes
    - S are geometric source terms
    - connection are Christoffel symbol contributions
    """

    def __init__(self, eos, atmosphere: Optional[AtmosphereParams] = None,
                 boundary_mode: str = "parity"):
        """
        Initialize GRHD equations.

        Parameters
        ----------
        eos : IdealGasEOS, PolytropicEOS
            Equation of state
        atmosphere : AtmosphereParams, optional
            Atmosphere configuration
        boundary_mode : str
            "parity" for reflecting at r=0, "outflow" for zero-gradient
        """
        self.eos = eos
        self.atmosphere = atmosphere if atmosphere is not None else AtmosphereParams()
        self.boundary_mode = boundary_mode

    def compute_rhs(self, D: np.ndarray, S_tildeD: np.ndarray, tau: np.ndarray,
                    rho0: np.ndarray, v_U: np.ndarray, pressure: np.ndarray,
                    W: np.ndarray, h: np.ndarray,
                    geometry: ValenciaGeometry,
                    bssn_vars, bssn_d1, background,
                    reconstructor, riemann_solver,
                    spacetime_mode: str, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute complete RHS for GRHD evolution equations.

        This is the main orchestration method that:
        1. Reconstructs primitives to cell faces
        2. Solves Riemann problem for fluxes
        3. Computes flux divergence
        4. Adds connection term contributions
        5. Adds geometric source terms

        Parameters
        ----------
        D, S_tildeD, tau : np.ndarray
            Conservative variables
        rho0, v_U, pressure, W, h : np.ndarray
            Primitive variables and derived quantities
        geometry : ValenciaGeometry
            Geometric quantities
        bssn_vars, bssn_d1 : BSSN variables and derivatives
        background : Background metric
        reconstructor : Reconstruction method
        riemann_solver : Riemann solver
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'
        r : np.ndarray
            Radial coordinate

        Returns
        -------
        rhs_D, rhs_S, rhs_tau : np.ndarray
            Right-hand sides for conservative evolution
        """
        # Step 1: Reconstruct primitives and convert to conservatives
        UL, UR, primL, primR = self.reconstruct_and_convert(
            rho0, v_U[:, 0], pressure, r, reconstructor, geometry
        )

        # Step 2: Solve Riemann problem and get fluxes
        F_D, F_S, F_tau = self.solve_riemann_and_densitize(
            UL, UR, primL, primR, geometry, riemann_solver, spacetime_mode
        )

        # Step 3: Compute divergence of fluxes

        # Step 3.1: Compute flux divergence (derivative term)
        rhs_D, rhs_S, rhs_tau = self.compute_divergence(F_D, F_S, F_tau, geometry.dr)

        # Step 3.2: Add connection term contributions from divergence
        if spacetime_mode != 'fixed_minkowski':
            conn_D, conn_S, conn_tau = self.compute_connection_terms(
                rho0, v_U, pressure, W, h, geometry, background.hat_christoffel
            )
            rhs_D += conn_D
            rhs_S += conn_S
            rhs_tau += conn_tau

        # Step 4: Add geometric source terms
        if spacetime_mode == 'dynamic':
            src_S, src_tau = self.compute_source_terms(
                rho0, v_U, pressure, W, h, geometry,
                bssn_vars, bssn_d1, background, spacetime_mode, r
            )
            rhs_S += src_S
            rhs_tau += src_tau

        return rhs_D, rhs_S, rhs_tau

    def compute_divergence(self, F_D_face: np.ndarray, F_S_face: np.ndarray,
                          F_tau_face: np.ndarray, dr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute flux divergence, derivative term.

        Formula: -∂_r(F̃^r) = -(F̃^r_{i+1/2} - F̃^r_{i-1/2}) / Δr

        Parameters
        ----------
        F_D_face : np.ndarray
            Density flux at cell faces (N_faces,)
        F_S_face : np.ndarray
            Momentum flux at cell faces (N_faces, 3)
        F_tau_face : np.ndarray
            Energy flux at cell faces (N_faces,)
        dr : float
            Radial grid spacing

        Returns
        -------
        rhs_D, rhs_S, rhs_tau : np.ndarray
            RHS contributions from flux divergence
        """
        # Determine number of cells from number of faces
        N = F_D_face.shape[0] + 1

        # Initialize RHS arrays
        rhs_D = np.zeros(N)
        rhs_S = np.zeros((N, SPACEDIM))
        rhs_tau = np.zeros(N)

        # Compute divergence using vectorized operations for interior cells
        inv_dr = 1.0 / (dr + 1e-30)

        # Interior cells: NUM_GHOSTS <= i < N - NUM_GHOSTS
        i_start = NUM_GHOSTS
        i_end = N - NUM_GHOSTS

        rhs_D[i_start:i_end]    = -(F_D_face[i_start:i_end]    -  F_D_face[i_start-1:i_end-1])    * inv_dr
        rhs_S[i_start:i_end, :] = -(F_S_face[i_start:i_end, :] -  F_S_face[i_start-1:i_end-1, :]) * inv_dr
        rhs_tau[i_start:i_end]  = -(F_tau_face[i_start:i_end]  -  F_tau_face[i_start-1:i_end-1])  * inv_dr

        return rhs_D, rhs_S, rhs_tau

    def reconstruct_and_convert(self, rho0: np.ndarray, v_primary: np.ndarray,
                                pressure: np.ndarray, r: np.ndarray,
                                reconstructor, geometry: ValenciaGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct primitive variables and convert to conservatives at cell faces.

        Parameters
        ----------
        rho0 : np.ndarray
            Rest mass density at cell centers
        v_primary : np.ndarray
            Primary velocity component (v^r in spherical)
        pressure : np.ndarray
            Pressure at cell centers
        r : np.ndarray
            Radial coordinate array
        reconstructor : Reconstruction
            Reconstruction method object
        geometry : ValenciaGeometry
            Geometry with metric at faces

        Returns
        -------
        UL_batch, UR_batch : np.ndarray
            Left/right conservative states at faces (N_faces, 3)
        primL_batch, primR_batch : np.ndarray
            Left/right primitive states at faces (N_faces, 3)
        """
        # Determine boundary condition type for reconstruction
        recon_boundary = "reflecting" if self.boundary_mode == "parity" else "open"

        # Reconstruct primitives to cell interfaces
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, v_primary, pressure, x=r, boundary_type=recon_boundary
        )

        # Extract interior faces (exclude ghost zones)
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]

        # Enforce reflecting condition at the r≈0 interface
        if recon_boundary == "reflecting" and (len(vL) > 0):
            # Enforce v=0 at the innermost face
            vL[0] = 0.0
            vR[0] = 0.0
            # For density and pressure, use symmetric values
            if len(rho0) > NUM_GHOSTS:
                rho_ref = rho0[NUM_GHOSTS]
                p_ref = pressure[NUM_GHOSTS]
                rhoL[0] = rho_ref
                rhoR[0] = rho_ref
                pL[0] = p_ref
                pR[0] = p_ref

        # Apply physical limiters if available
        if hasattr(reconstructor, "apply_physical_limiters"):
            # Get metric at faces (primary direction)
            gamma_rr_f = geometry.gamma_LL[:, 0, 0] if hasattr(geometry.gamma_LL, 'shape') else np.ones(len(rhoL))
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere=self.atmosphere,
                gamma_rr=gamma_rr_f
            )

        # Get metric component at faces for cons conversion
        gamma_primary_primary_f = geometry.gamma_LL[:, 0, 0] if hasattr(geometry.gamma_LL, 'shape') else np.ones(len(rhoL))

        # Convert primitives to conservatives at interfaces
        UL_D, UL_S_primary, UL_tau = prim_to_cons(
            rhoL, vL, pL, gamma_primary_primary_f, self.eos
        )
        UR_D, UR_S_primary, UR_tau = prim_to_cons(
            rhoR, vR, pR, gamma_primary_primary_f, self.eos
        )

        # Package for Riemann solver
        UL_batch = np.stack([UL_D, UL_S_primary, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_S_primary, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        return UL_batch, UR_batch, primL_batch, primR_batch

    def solve_riemann_and_densitize(self, UL_batch: np.ndarray, UR_batch: np.ndarray,
                                    primL_batch: np.ndarray, primR_batch: np.ndarray,
                                    geometry: ValenciaGeometry, riemann_solver,
                                    spacetime_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Riemann problem and apply densitization.

        Parameters
        ----------
        UL_batch, UR_batch : np.ndarray
            Conservative states at faces
        primL_batch, primR_batch : np.ndarray
            Primitive states at faces
        geometry : ValenciaGeometry
            Geometry with lapse, shift, metric at faces
        riemann_solver : RiemannSolver
            Riemann solver object
        spacetime_mode : str
            'fixed_minkowski', 'fixed', or 'dynamic'

        Returns
        -------
        F_D_face, F_S_tildeD_face, F_tau_face : np.ndarray
            Densitized fluxes at cell faces
        """
        # Extract face geometry (for now use cell-center values)
        alpha_f = geometry.alpha
        beta_primary_f = geometry.beta_U[:, 0]
        gamma_primary_primary_f = geometry.gamma_LL[:, 0, 0]
        e6phi_f = geometry.e6phi

        # Solve Riemann problem to get physical fluxes
        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_primary_primary_f, alpha_f, beta_primary_f, self.eos
        )

        # Densitization: F̃ = α e^{6φ} F^phys
        dens_factor = alpha_f * e6phi_f
        F_batch = dens_factor[:, None] * F_phys_batch

        # Enforce zero momentum flux at the r≈0 interface (spherical boundary)
        if self.boundary_mode == "parity" and spacetime_mode != "fixed_minkowski":
            if len(F_batch) > 0:
                F_batch[0, 1] = 0.0  # F_Sr = 0 at the r≈0 interface

        # Additional freeze: zero fluxes at atmosphere interfaces
        if spacetime_mode != "fixed_minkowski":
            try:
                rho_atm_face = self.atmosphere.rho_floor * getattr(self.atmosphere, 'rho_threshold', 1.0)
                rhoL = primL_batch[:, 0]
                rhoR = primR_batch[:, 0]
                mask_freeze = (rhoL < rho_atm_face) | (rhoR < rho_atm_face)
                if np.any(mask_freeze):
                    F_batch[mask_freeze, :] = 0.0
            except Exception:
                pass

        # Construct 3D momentum flux array
        N_faces = len(F_batch)
        F_S_tildeD_face = np.zeros((N_faces, SPACEDIM))
        F_S_tildeD_face[:, 0] = F_batch[:, 1]  # Primary direction flux

        return F_batch[:, 0], F_S_tildeD_face, F_batch[:, 2]

    def compute_connection_terms(self, rho0: np.ndarray, v_U: np.ndarray,
                                pressure: np.ndarray, W: np.ndarray, h: np.ndarray,
                                geometry: ValenciaGeometry,
                                hat_chris: np.ndarray,
                                return_debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Christoffel connection contributions to RHS.

        Connection form (NRPy+):
            D, τ:   -Γ̂^k_{ki} F̃^i
            S_i:    -Γ̂^k_{kj} F̃^j_i + Γ̂^l_{ji} F̃^j_l

        Parameters
        ----------
        rho0, v_U, pressure, W, h : np.ndarray
            Hydrodynamic variables
        geometry : ValenciaGeometry
            Geometric quantities
        hat_chris : np.ndarray
            Reference Christoffel symbols (N, 3, 3, 3)

        Returns
        -------
        conn_D, conn_S, conn_tau : np.ndarray
            Connection contributions to RHS
        """
        N = len(rho0)
        alpha = geometry.alpha
        e6phi = geometry.e6phi
        gamma_LL = geometry.gamma_LL
        gamma_UU = geometry.gamma_UU

        # Compute 4-velocity components
        u4U = np.zeros((N, 4))
        u4U[:, 0] = W / alpha  # u^0
        for i in range(SPACEDIM):
            u4U[:, i+1] = W * v_U[:, i] - geometry.beta_U[:, i] * u4U[:, 0]  # u^i

        # Compute stress-energy tensor components needed for fluxes
        factor = rho0 * h * W**2

        # Conservative density flux
        rho_star = alpha * e6phi * rho0 * u4U[:, 0]

        # Physical fluxes at cell centers (partial flux vectors)
        fD_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fD_U[:, j] = rho_star * v_U[:, j]

        # Energy (tau) partial flux vector
        # Compute T^0j first
        T0j = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            T0j[:, j] = factor * u4U[:, 0] * u4U[:, j+1]

        fTau_U = np.zeros((N, SPACEDIM))
        for j in range(SPACEDIM):
            fTau_U[:, j] = alpha ** 2 * e6phi * T0j[:, j] - rho_star * v_U[:, j]

        # Momentum partial flux tensor F̃^j_i = α e^{6φ} T^j_i
        # First compute T^ij
        Tij_UU = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                Tij_UU[:, i, j] = (factor * u4U[:, i+1] * u4U[:, j+1] +
                                   pressure * gamma_UU[:, i, j])

        # Convert to mixed T^j_i
        Tij_UD = np.zeros((N, SPACEDIM, SPACEDIM))
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                for k in range(SPACEDIM):
                    Tij_UD[:, j, i] += Tij_UU[:, j, k] * gamma_LL[:, i, k]

        F_S_no_ghat = np.zeros((N, SPACEDIM, SPACEDIM))
        for j in range(SPACEDIM):
            for i in range(SPACEDIM):
                F_S_no_ghat[:, j, i] = alpha * e6phi * Tij_UD[:, j, i]

        # Connection contributions
        Gamma_trace = np.einsum('xkkj->xj', hat_chris)
        conn_D = -np.einsum('xj,xj->x', Gamma_trace, fD_U)
        conn_tau = -np.einsum('xj,xj->x', Gamma_trace, fTau_U)
        conn_S = (
            -np.einsum('xl,xli->xi', Gamma_trace, F_S_no_ghat)
            + np.einsum('xlji,xjl->xi', hat_chris, F_S_no_ghat)
        )

        if not return_debug:
            return conn_D, conn_S, conn_tau

        debug = {
            "rho_star_flux": fD_U.copy(),
            "tau_flux": fTau_U.copy(),
            "momentum_flux": F_S_no_ghat.copy(),
            "Gamma_trace": Gamma_trace.copy(),
        }
        return conn_D, conn_S, conn_tau, debug

    def compute_source_terms(self, rho0: np.ndarray, v_U: np.ndarray,
                            pressure: np.ndarray, W: np.ndarray, h: np.ndarray,
                            geometry: ValenciaGeometry,
                            bssn_vars, bssn_d1, background,
                            spacetime_mode: str, r: np.ndarray,
                            return_debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute geometric source terms for Valencia equations.

        Source terms arise from:
        1. Extrinsic curvature K_ij contractions
        2. Lapse gradient ∂_i α
        3. Shift covariant derivative ∇̂_i β^j
        4. Metric covariant derivative ∇̂_i γ_{jk}

        Parameters
        ----------
        rho0, v_U, pressure, W, h : np.ndarray
            Hydrodynamic variables
        geometry : ValenciaGeometry
            Geometric quantities
        bssn_vars, bssn_d1 : BSSN variables and derivatives
        background : Background metric
        spacetime_mode : str
            'fixed' or 'dynamic'
        r : np.ndarray
            Radial coordinate

        Returns
        -------
        src_S, src_tau : np.ndarray
            Source terms for momentum and energy
        """
        N = len(rho0)

        # Fixed spacetime has no source terms
        if spacetime_mode == 'fixed':
            src_S_vector = np.zeros((N, SPACEDIM))
            src_tau = np.zeros(N)
            if not return_debug:
                return src_S_vector, src_tau

            zero_term_vec = np.zeros((N, SPACEDIM))
            zero_term_scalar = np.zeros(N)
            zero_hat = np.zeros((N, SPACEDIM, SPACEDIM))
            debug_fixed = {
                "energy_Kij": zero_term_scalar.copy(),
                "energy_dalpha": zero_term_scalar.copy(),
                "energy_total": zero_term_scalar.copy(),
                "momentum_T00_alpha": zero_term_vec.copy(),
                "momentum_T0j_beta": zero_term_vec.copy(),
                "momentum_metric": zero_term_vec.copy(),
                "momentum_total": zero_term_vec.copy(),
                "hatD_beta": zero_hat.copy(),
            }
            return src_S_vector, src_tau, debug_fixed

        # Extract needed quantities
        alpha = geometry.alpha
        beta_U = geometry.beta_U
        gamma_LL = geometry.gamma_LL
        gamma_UU = geometry.gamma_UU
        e6phi = geometry.e6phi

        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)

        # Create stress-energy tensor object for cleaner code
        stress_energy = StressEnergyTensor(
            geometry=ADMGeometry(alpha, beta_U, gamma_LL, gamma_UU),
            rho0=rho0, v_U=v_U, pressure=pressure, W=W, h=h
        )

        # Compute stress-energy tensor components
        T00, T0i, Tij = stress_energy.compute_T4UU()

        # For source terms, we also need mixed components T^0_i
        T0i_UD = np.zeros((N, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                T0i_UD[:, i] += T0i[:, j] * gamma_LL[:, i, j]

        # ====================================================================
        # ENERGY SOURCE TERM
        # ====================================================================
        # S_τ = α e^{6φ} [K_ij stress_block - T^0_μ ∂_μ α]

        # Get BSSN quantities
        if hasattr(bssn_vars, 'hDD'):
            h_LL = bssn_vars.hDD
        elif hasattr(bssn_vars, 'h_LL'):
            h_LL = bssn_vars.h_LL
        else:
            h_LL = get_bar_gamma_LL(bssn_vars, background)

        K = np.asarray(bssn_vars.K, dtype=float)

        if hasattr(bssn_vars, 'aDD'):
            a_LL = bssn_vars.aDD
        elif hasattr(bssn_vars, 'a_LL'):
            a_LL = bssn_vars.a_LL
        else:
            a_LL = get_bar_A_LL(bssn_vars)

        # Physical extrinsic curvature: K_ij = e^{4φ} Ā_ij + (1/3) γ_ij K
        K_LL = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                K_LL[:, i, j] = e4phi * a_LL[:, i, j] + (1.0/3.0) * gamma_LL[:, i, j] * K

        # Stress block: T^00 β^i β^j + 2 T^{0i} β^j + T^{ij}
        stress_block = np.zeros((N, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                stress_block[:, i, j] = (
                    T00 * beta_U[:, i] * beta_U[:, j] +
                    2.0 * T0i[:, i] * beta_U[:, j] +
                    Tij[:, i, j]
                )

        # Term 1: K_ij × stress_block
        term1_tau = np.einsum('nij,nij->n', K_LL, stress_block)

        # Term 2: Lapse gradient
        d_alpha = np.asarray(bssn_d1.lapse, dtype=float)
        term2_tau = np.zeros(N)
        for i in range(SPACEDIM):
            term2_tau -= (T00 * beta_U[:, i] + T0i[:, i]) * d_alpha[:, i]

        src_tau = alpha * e6phi * (term1_tau + term2_tau)

        # ====================================================================
        # MOMENTUM SOURCE TERM
        # ====================================================================
        # S_i = α e^{6φ} [-T^{00} ∂_i α + T^0_j ∇̂_i β^j + 0.5 stress_block × ∇̂_i γ_{jk}]

        # Prepare momentum term containers
        first_term = -T00[:, None] * alpha[:, None] * d_alpha
        second_term = np.zeros((N, SPACEDIM))
        third_term = np.zeros((N, SPACEDIM))

        # Term 2: T^0_j ∇̂_i β^j (shift covariant derivative)
        # ∇̂_i β^j = ∂_i β^j + Γ̂^j_{ik} β^k
        d_beta = np.asarray(bssn_d1.shift_U, dtype=float)
        hat_chris = background.hat_christoffel
        hatD_beta = np.zeros((N, SPACEDIM, SPACEDIM))

        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                cov_deriv_beta = d_beta[:, j, i]
                for k in range(SPACEDIM):
                    cov_deriv_beta += hat_chris[:, j, i, k] * beta_U[:, k]
                hatD_beta[:, i, j] = cov_deriv_beta
                second_term[:, i] += T0i_UD[:, j] * cov_deriv_beta

        # Term 3: 0.5 × stress_block × ∇̂_i γ_{jk}
        # ∇̂_i γ_{jk} = e^{4φ} (∂_i γ̄_{jk} + 4 γ̄_{jk} ∂_i φ)
        # where ∂_i γ̄_{jk} = Γ̂_{jik} + Γ̂_{kij} from reference metric
        d_phi = np.asarray(bssn_d1.phi, dtype=float)

        if hasattr(background, 'hat_D_gamma'):
            hat_D_gamma = background.hat_D_gamma
        else:
            hat_D_gamma = get_hat_D_bar_gamma_LL(h_LL, hat_chris)

        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    cov_deriv_gamma = e4phi * (hat_D_gamma[:, i, j, k] +
                                               4.0 * h_LL[:, j, k] * d_phi[:, i])
                    third_term[:, i] += 0.5 * stress_block[:, j, k] * cov_deriv_gamma

        total_momentum_term = first_term + second_term + third_term
        src_S_vector = alpha[:, None] * e6phi[:, None] * total_momentum_term

        if not return_debug:
            return src_S_vector, src_tau

        debug = {
            "energy_Kij": term1_tau.copy(),
            "energy_dalpha": term2_tau.copy(),
            "energy_total": src_tau.copy(),
            "momentum_T00_alpha": first_term.copy(),
            "momentum_T0j_beta": second_term.copy(),
            "momentum_metric": third_term.copy(),
            "momentum_total": src_S_vector.copy(),
            "hatD_beta": hatD_beta.copy(),
        }
        return src_S_vector, src_tau, debug
