# valencia_reference_metric.py
"""
Valencia with reference metric - explicit connection terms.

Computes: ∂_r F + F · Γ̂^k_{rk} explicitly instead of using Engrenage divergence.
"""

import numpy as np

from source.bssn.tensoralgebra import (
    SPACEDIM,
    get_bar_gamma_LL,
    get_det_bar_gamma,
    get_bar_A_LL,
    get_hat_D_bar_gamma_LL
)
from source.backgrounds.sphericalbackground import i_r, i_t, i_p
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons


class ValenciaReferenceMetric:
    """Valencia with reference metric - explicit derivative + connection form."""

# valencia_reference_metric.py (solo la parte modificada)

    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        RHS with explicit form: ∂_r F + F · Γ̂^k_{rk}
        """
        
        dr = self._get_mesh_spacing(grid, r)
        N = len(r)
        
        D, Sr, tau = D.copy(), Sr.copy(), tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)
        
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)
        
        flux_phys = self._compute_interface_fluxes(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )
        
        # ====================================================================
        # COMPUTE Γ̂^k_{rk} ANALYTICALLY from Christoffel symbols
        # ====================================================================
        # Γ̂^k_{rk} = Γ̂^r_{rr} + Γ̂^θ_{rθ} + Γ̂^φ_{rφ}
        # Using hat_christoffel[x, i, j, k] = Γ̂^i_{jk}
        
        hat_chris = background.hat_christoffel  # (N, i, j, k)
        
        Gamma_trace = (
            hat_chris[:, i_r, i_r, i_r]  # Γ̂^r_{rr}
            + hat_chris[:, i_t, i_r, i_t]  # Γ̂^θ_{rθ}
            + hat_chris[:, i_p, i_r, i_p]  # Γ̂^φ_{rφ}
        )
        
        # For spherical: = 0 + 1/r + 1/r = 2/r
        
        # Fluxes at faces
        F_D_face = flux_phys['D']
        F_Sr_face = flux_phys['Sr']
        F_tau_face = flux_phys['tau']
        
        # Interpolate fluxes to cell centers for connection term
        F_D_cell = np.zeros(N)
        F_Sr_cell = np.zeros(N)
        F_tau_cell = np.zeros(N)
        
        F_D_cell[1:-1] = 0.5 * (F_D_face[:-1] + F_D_face[1:])
        F_Sr_cell[1:-1] = 0.5 * (F_Sr_face[:-1] + F_Sr_face[1:])
        F_tau_cell[1:-1] = 0.5 * (F_tau_face[:-1] + F_tau_face[1:])
        
        # ====================================================================
        # MASS: ∂_t(J D) = -∂_r F_D - F_D · Γ̂^k_{rk}
        # ====================================================================
        
        rhs_D = np.zeros(N)
        rhs_D[1:-1] = -(F_D_face[1:] - F_D_face[:-1]) / dr
        rhs_D[1:-1] += -F_D_cell[1:-1] * Gamma_trace[1:-1]
        
        # ====================================================================
        # ENERGY: ∂_t(J τ) = -∂_r F_τ - F_τ · Γ̂^k_{rk}
        # ====================================================================
        
        rhs_tau = np.zeros(N)
        rhs_tau[1:-1] = -(F_tau_face[1:] - F_tau_face[:-1]) / dr
        rhs_tau[1:-1] += -F_tau_cell[1:-1] * Gamma_trace[1:-1]
        
        # ====================================================================
        # MOMENTUM: ∂_t(J S_r) = -∂_r F_Sr - F_Sr · Γ̂^k_{rk} + α J p · Γ̂^k_{rk}
        # ====================================================================
        
        rhs_Sr = np.zeros(N)
        rhs_Sr[1:-1] = -(F_Sr_face[1:] - F_Sr_face[:-1]) / dr
        rhs_Sr[1:-1] += -F_Sr_cell[1:-1] * Gamma_trace[1:-1]
        rhs_Sr[1:-1] += (g['alpha'][1:-1] * g['J_cell'][1:-1] * 
                        pressure[1:-1] * Gamma_trace[1:-1])
        
        # ====================================================================
        # PHYSICAL SOURCES
        # ====================================================================
        
        src_Sr, src_tau = self._compute_source_terms(
            rho0, vr, pressure, W, h, g, bssn_vars, bssn_d1,
            background, spacetime_mode, r
        )
        
        rhs_Sr += src_Sr
        rhs_tau += src_tau
        
        # ====================================================================
        # DIVIDE BY J
        # ====================================================================
        
        J = g['J_cell'] + 1e-30
        rhs_D /= J
        rhs_Sr /= J
        rhs_tau /= J
        
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(
            rhs_D, rhs_Sr, rhs_tau, r
        )
        
        return rhs_D, rhs_Sr, rhs_tau
    
    
    # ========================================================================
    # SOURCES (same as before - correct with einsum)
    # ========================================================================
    
    def _compute_source_terms(self, rho0, vr, pressure, W, h, g,
                              bssn_vars, bssn_d1, background, spacetime_mode, r):
        """Physical sources with einsum."""
        
        N = len(r)
        J = g['J_cell']
        
        if spacetime_mode == "fixed_minkowski":
            return np.zeros(N), np.zeros(N)
        
        alpha, beta_r, gamma_rr = g['alpha'], g['beta_r'], g['gamma_rr']
        
        T00, T0r, Trr = self._compute_stress_energy_tensor(
            rho0, vr, pressure, W, h, alpha, beta_r, gamma_rr
        )
        
        # Build 3D tensors
        beta_U = np.zeros((N, SPACEDIM))
        beta_U[:, i_r] = beta_r
        
        T0U = np.zeros((N, SPACEDIM))
        T0U[:, i_r] = T0r
        
        phi = np.asarray(bssn_vars.phi, dtype=float)
        e4phi = np.exp(4.0 * phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        gamma_LL = e4phi[:, None, None] * bar_gamma_LL
        gamma_UU = np.linalg.inv(gamma_LL)
        
        TUU = np.zeros((N, SPACEDIM, SPACEDIM))
        TUU[:, i_r, i_r] = Trr
        TUU[:, i_t, i_t] = pressure * gamma_UU[:, i_t, i_t]
        TUU[:, i_p, i_p] = pressure * gamma_UU[:, i_p, i_p]
        
        beta_lower = np.einsum('xij,xj->xi', gamma_LL, beta_U)
        T0_lower = T00[:, None] * beta_lower + np.einsum('xj,xij->xi', T0U, gamma_LL)
        
        # Geometric derivatives
        dalpha_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            dalpha_dx = np.asarray(bssn_d1.lapse)
        
        Dhat_beta_U = np.zeros((N, SPACEDIM, SPACEDIM))
        if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3:
                Dhat_beta_U = shift_d1.copy()
                hat_chris = background.hat_christoffel
                Dhat_beta_U += np.einsum('xjik,xk->xij', hat_chris, beta_U)
        
        dphi_dx = np.zeros((N, SPACEDIM))
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            dphi_dx = np.asarray(bssn_d1.phi)
        
        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL,
                                                  bssn_d1.h_LL, background)
        
        Dhat_gamma_LL = np.zeros((N, SPACEDIM, SPACEDIM, SPACEDIM))
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                for k in range(SPACEDIM):
                    Dhat_gamma_LL[:, i, j, k] = e4phi * (
                        4.0 * bar_gamma_LL[:, j, k] * dphi_dx[:, i]
                        + hat_D_bar_gamma[:, j, k, i]
                    )
        
        K = np.asarray(bssn_vars.K, dtype=float)
        bar_A_LL = get_bar_A_LL(r, bssn_vars, background)
        K_LL = e4phi[:, None, None] * bar_A_LL + (K / 3.0)[:, None, None] * gamma_LL
        
        tensor_block = (
            T00[:, None, None] * np.einsum('xi,xj->xij', beta_U, beta_U)
            + 2.0 * np.einsum('xi,xj->xij', T0U, beta_U)
            + TUU
        )
        
        # Sources with einsum contractions
        src_Sr_vector = J[:, None] * alpha[:, None] * (
            -T00[:, None] * alpha[:, None] * dalpha_dx
            + np.einsum('xi,xij->xi', T0_lower, Dhat_beta_U)
            + 0.5 * np.einsum('xjk,xijk->xi', tensor_block, Dhat_gamma_LL)
        )
        
        src_tau = J * alpha * (
            np.einsum('x,xi,xj,xij->x', T00, beta_U, beta_U, K_LL)
            - np.einsum('x,xi,xi->x', T00, beta_U, dalpha_dx)
            + 2.0 * np.einsum('xi,xj,xij->x', T0U, beta_U, K_LL)
            - np.einsum('xi,xi->x', T0U, dalpha_dx)
            + np.einsum('xij,xij->x', TUU, K_LL)
        )
        
        return src_Sr_vector[:, i_r], src_tau

    # ========================================================================
    # GEOMETRY & UTILITIES
    # ========================================================================
    
    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        N = len(r)
        g = {}
        
        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            g['beta_r'] = np.zeros(N)
            g['e6phi'] = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)
            
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                g['beta_r'] = (shift_array[:, i_r].astype(float)
                              if shift_array.ndim >= 2 else np.zeros(N))
            else:
                g['beta_r'] = np.zeros(N)
            
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            g['e6phi'] = np.exp(6.0 * phi_arr)
            
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            e4phi = np.exp(4.0 * phi_arr)
            g['gamma_rr'] = e4phi * bar_gamma_LL[:, i_r, i_r]
        
        if spacetime_mode != "fixed_minkowski":
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            sqrt_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)
            sqrt_hat_gamma = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            J_cell = g['e6phi'] * sqrt_bar_gamma / sqrt_hat_gamma
        else:
            J_cell = np.ones(N)
        
        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])
        
        return g
    
    def _compute_interface_fluxes(self, rho0, vr, pressure, g, r,
                                   eos, reconstructor, riemann_solver):
        """
        Compute physical fluxes (α J F_phys) WITHOUT √ĝ factor.
        
        Returns fluxes at faces ready for ∂_r F calculation.
        """
        N = len(r)
        
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )
        
        rhoL, vL, pL = rhoL[1:-1], vL[1:-1], pL[1:-1]
        rhoR, vR, pR = rhoR[1:-1], vR[1:-1], pR[1:-1]
        
        alpha_f = 0.5 * (g['alpha'][:-1] + g['alpha'][1:])
        beta_r_f = 0.5 * (g['beta_r'][:-1] + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f = g['J_face']
        
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999999,
                gamma_rr=gamma_rr_f
            )
        
        UL_D, UL_Sr, UL_tau = prim_to_cons(rhoL, vL, pL, gamma_rr_f, eos)
        UR_D, UR_Sr, UR_tau = prim_to_cons(rhoR, vR, pR, gamma_rr_f, eos)
        
        UL_batch = np.stack([UL_D, UL_Sr, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_Sr, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)
        
        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_rr_f, alpha_f, beta_r_f, eos
        )
        
        # Multiply by (α J) only, NOT by √ĝ
        dens_factor = alpha_f * J_f
        F_batch = dens_factor[:, None] * F_phys_batch
        
        return {
            'D': F_batch[:, 0],
            'Sr': F_batch[:, 1],
            'tau': F_batch[:, 2]
        }
    
    def _compute_stress_energy_tensor(self, rho0, vr, pressure, W, h,
                                      alpha, beta_r, gamma_rr):
        grr = 1.0 / gamma_rr
        beta_u = grr * beta_r
        ut = W / alpha
        ur = W * (vr - beta_u / alpha)
        
        g00 = -1.0 / (alpha ** 2)
        g0r = beta_u / (alpha ** 2)
        grr_eff = grr - (beta_u ** 2) / (alpha ** 2)
        
        T00 = rho0 * h * ut * ut + pressure * g00
        T0r = rho0 * h * ut * ur + pressure * g0r
        Trr = rho0 * h * ur * ur + pressure * grr_eff
        
        return T00, T0r, Trr
    
    def _get_mesh_spacing(self, grid, r):
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            return float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            return float(grid.dr)
        return float(r[1] - r[0]) if len(r) > 1 else 1.0
    
    def _apply_ghost_cell_boundaries(self, D, Sr, tau, r):
        N = len(r)
        if NUM_GHOSTS > 0:
            mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            D[:NUM_GHOSTS], Sr[:NUM_GHOSTS], tau[:NUM_GHOSTS] = D[mir], -Sr[mir], tau[mir]
            last = N - NUM_GHOSTS - 1
            if last >= 0:
                D[-NUM_GHOSTS:], Sr[-NUM_GHOSTS:], tau[-NUM_GHOSTS:] = D[last], Sr[last], tau[last]
        return D, Sr, tau
    
    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        N = len(r)
        if NUM_GHOSTS > 0:
            mir = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            rhs_D[:NUM_GHOSTS], rhs_Sr[:NUM_GHOSTS], rhs_tau[:NUM_GHOSTS] = rhs_D[mir], -rhs_Sr[mir], rhs_tau[mir]
            last = N - NUM_GHOSTS - 1
            if last >= 0:
                rhs_D[-NUM_GHOSTS:], rhs_Sr[-NUM_GHOSTS:], rhs_tau[-NUM_GHOSTS:] = rhs_D[last], rhs_Sr[last], rhs_tau[last]
        return rhs_D, rhs_Sr, rhs_tau
    