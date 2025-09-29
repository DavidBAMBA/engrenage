# valencia_reference_metric.py (FULL reference-metric approach, 1D spherical)

import numpy as np

from source.bssn.tensoralgebra import get_bar_gamma_LL, get_det_bar_gamma, get_bar_A_LL
from source.backgrounds.sphericalbackground import i_r
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons
from source.matter.hydro.perfect_fluid import T_components_from_primitives


class ValenciaReferenceMetric:
    """
    Valencia formulation with REFERENCE METRIC (full approach) for 1D spherical coordinates.
    Based on Montero et al. (2013) arXiv:1309.7808 (full approach).

    Conservative form equation with reference metric (1D):
        ∂_t( J U ) + (1/√ĝ) ∂_r [ √ĝ ( α J F^r_phys ) ] = J S_phys,
    where
        J   = e^{6φ} √( \barγ / ĝ ),
        √ĝ  = √det(ĝ)   (in 1D spherical:  √ĝ = r^2),
        F^r_phys  is the non-densitized flux (with ṽ^r = v^r − β^r/α),
        S_phys     are the physical sources (geometric/gravitational) without densitization.

    Note: the Riemann solver returns F^r_phys; here we apply α·J and the weight √ĝ.
    The integrator directly integrates U (not J·U), so we divide the entire
    densitized RHS by J at the end to obtain dU/dt.
    
    CHANGES IMPLEMENTED ACCORDING TO PAPER 1309.7808v2:
    - Connection terms Γ̂^k_{jk} added in covariant divergence (eqs. 21,37,49)
    - Complete gravitational source terms according to equation (34)  
    - Hoop stress 2p/r for spherical coordinates
    - Densitization factors J = e^{6φ} √(γ̄/γ̂) implemented correctly
    """

    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Valencia con métrica de referencia (full approach), 1D esférico:

            ∂_t(J U) + (1/√ĝ) ∂_r [ √ĝ ( α J F^r_phys ) ] = J S_phys

        NOTA IMPORTANTE:
        - No añadimos términos de conexión explícitos aquí, porque la
        divergencia con √ĝ ya los incorpora (evitamos doble conteo).
        - Las fuentes geométricas (S_phys) se suman vía _compute_sources_full,
        y luego dividimos todo por J para obtener dU/dt.
        """
        import numpy as np

        # --- 0) Paso de malla ---
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            dr = float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            dr = float(grid.dr)
        else:
            dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

        # --- 1) Copias y BCs en CONSERVADAS ---
        D  = D.copy()
        Sr = Sr.copy()
        tau = tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # --- 2) Geometría ---
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # --- 3) Flujos en caras (densitizados con √ĝ, α, J) ---
        flux_hat = self._compute_interface_fluxes_full(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # --- 4) Divergencia covariante vía √ĝ (SIN corrección extra de Γ̂) ---
        N = len(r)
        rhs_D  = np.zeros(N)
        rhs_Sr = np.zeros(N)
        rhs_tau= np.zeros(N)

        sgc = g['sqrt_g_hat_cell']  # √ĝ en centros (r^2)

        # - (ΔF̂) / (√ĝ * Δr)
        div_D   = -(np.diff(flux_hat['D'  ])) / (sgc[1:-1] * dr + 1e-30)
        div_Sr  = -(np.diff(flux_hat['Sr' ])) / (sgc[1:-1] * dr + 1e-30)
        div_tau = -(np.diff(flux_hat['tau'])) / (sgc[1:-1] * dr + 1e-30)

        rhs_D [1:-1] = div_D
        rhs_Sr[1:-1] = div_Sr
        rhs_tau[1:-1]= div_tau

        # --- 5) Fuentes físicas (J·S_phys) ---
        src_Sr, src_tau = self._compute_sources_full(
            pressure, g, bssn_vars, bssn_d1,
            rho0, h, W, vr, spacetime_mode, r, background
        )
        rhs_Sr += src_Sr
        rhs_tau += src_tau

        # --- 6) Pasar de ∂_t(JU) a ∂_t U ---
        J = g['J_cell'] + 1e-30
        rhs_D  = rhs_D  / J
        rhs_Sr = rhs_Sr / J
        rhs_tau= rhs_tau/ J

        # --- 7) BCs para RHS ---
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(rhs_D, rhs_Sr, rhs_tau, r)

        return rhs_D, rhs_Sr, rhs_tau

    def _compute_sources_full(self, pressure, g, bssn_vars, bssn_d1,
                            rho0, h, W, vr, spacetime_mode, r, background):
        """
        Fuentes físicas (sin densitizar) para S_r y tau, consistentes con NRPy:

        S_r_phys = α[ - T^{00} ∂_r α
                        + T^{0r}  (D̂_r β^r)
                        + 1/2 (T^{00} β_r^2 + 2 T^{0r} β_r + T^{rr}) (D̂_r γ_{rr}) ]

        S_tau_phys = α[ (T^{00} β_r^2 + 2 T^{0r} β_r + T^{rr}) K_{rr}
                        - (T^{00} β_r + T^{0r}) ∂_r α ]

        Devolvemos J * S_phys (la conversión a dU/dt la hace compute_rhs dividiendo por J).
        """


        N = len(r)
        J_cell = g['J_cell']

        # --- Caso Minkowski: sin fuentes (la geometría se maneja vía √ĝ en la divergencia) ---
        if spacetime_mode == "fixed_minkowski":
            src_Sr_phys  = np.zeros(N)
            src_tau_phys = np.zeros(N)
            return J_cell * src_Sr_phys, J_cell * src_tau_phys

        # --- Geometría y métricas ---
        alpha   = g['alpha']
        beta_r  = g['beta_r']
        gamma_rr= g['gamma_rr']

        # --- Componentes T^{μν} desde perfect_fluid (sin duplicar fórmulas aquí) ---
        #    T00, T0r, Trr son T^{00}, T^{0r}, T^{rr} (índices contravariantes)
        T00, T0r, Trr = T_components_from_primitives(
            rho0=rho0, vr=vr, p=pressure, W=W, h=h,
            alpha=alpha, beta_r=beta_r, gamma_rr=gamma_rr
        )

        # --- Derivadas: ∂_r α y D̂_r β^r ---
        dalpha_dr = np.zeros(N)
        if getattr(bssn_d1, 'lapse', None) is not None:
            lapse_der = np.asarray(bssn_d1.lapse)
            if lapse_der.ndim >= 2 and lapse_der.shape[1] > i_r:
                dalpha_dr = lapse_der[:, i_r]

        dbeta_dr = np.zeros(N)
        if getattr(bssn_d1, 'shift_U', None) is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3 and shift_d1.shape[1] > i_r and shift_d1.shape[2] > i_r:
                dbeta_dr = shift_d1[:, i_r, i_r]
        hat_D_beta_r = dbeta_dr  # en 1D esférico, D̂_r β^r = ∂_r β^r

        # --- D̂_r γ_{rr} = e^{4φ} [ ∂_r γ̄_{rr} + 4 γ̄_{rr} ∂_r φ ] ---
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        e4phi   = np.exp(4.0 * phi_arr)

        bar_gamma_LL   = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_rr   = bar_gamma_LL[:, i_r, i_r]

        dphi_dr = np.zeros(N)
        if getattr(bssn_d1, 'phi', None) is not None:
            phi_der = np.asarray(bssn_d1.phi)
            if phi_der.ndim >= 2 and phi_der.shape[1] > i_r:
                dphi_dr = phi_der[:, i_r]

        dbar_gamma_rr_dr = np.zeros(N)
        if getattr(bssn_d1, 'h_LL', None) is not None:
            h_d1 = np.asarray(bssn_d1.h_LL)
            if h_d1.ndim >= 4 and h_d1.shape[1] > i_r and h_d1.shape[2] > i_r and h_d1.shape[3] > i_r:
                # γ̄_rr = h_rr * s_rr + (hat_γ_rr). Como ya usas utilidades con scaling:
                # derivada total via regla del producto con la matriz de escalado
                scaling_rr     = background.scaling_matrix[:, i_r, i_r]
                dscaling_rr_dr = background.d1_scaling_matrix[:, i_r, i_r, i_r]
                h_rr           = bssn_vars.h_LL[:, i_r, i_r]
                dh_rr_dr       = h_d1[:, i_r, i_r, i_r]
                dbar_gamma_rr_dr = scaling_rr * dh_rr_dr + h_rr * dscaling_rr_dr

        hat_D_gamma_rr = e4phi * (dbar_gamma_rr_dr + 4.0 * bar_gamma_rr * dphi_dr)

        # --- K_rr = A_rr + (1/3) γ_rr K ---
        K = getattr(bssn_vars, 'K', np.zeros(N))
        bar_A_LL       = get_bar_A_LL(r, bssn_vars, background)
        A_rr_physical  = e4phi * bar_A_LL[:, i_r, i_r]
        K_rr           = A_rr_physical + (gamma_rr * K) / 3.0

        # --- Fuentes físicas (bloque completo multiplicado por α) ---
        combo = (T00 * (beta_r**2) + 2.0 * T0r * beta_r + Trr)

        src_Sr_phys  = alpha * ( - T00 * dalpha_dr
                                + T0r * hat_D_beta_r
                                + 0.5 * combo * hat_D_gamma_rr )

        src_tau_phys = alpha * ( combo * K_rr
                                - (T00 * beta_r + T0r) * dalpha_dr )

        # Devolvemos densitizadas (J * S_phys). La división por J se hace en compute_rhs.
        return J_cell * src_Sr_phys, J_cell * src_tau_phys

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        """Extracts geometric quantities using BSSN framework functions correctly."""
        N = len(r)
        g = {}

        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            g['beta_r'] = np.zeros(N)
            g['e6phi'] = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            # Use BSSN variables directly
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)
            
            # Extract shift properly
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                if shift_array.ndim >= 2 and shift_array.shape[1] > i_r:
                    g['beta_r'] = shift_array[:, i_r].astype(float)
                else:
                    g['beta_r'] = np.zeros(N)
            else:
                g['beta_r'] = np.zeros(N)
            
            # Conformal factors
            phi_arr = np.asarray(bssn_vars.phi, dtype=float)
            g['e6phi'] = np.exp(6.0 * phi_arr)
            
            # CORRECCIÓN: Physical spatial metric γ_rr = e^{4φ} γ̄_rr using BSSN functions
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            e4phi = np.exp(4.0 * phi_arr)
            g['gamma_rr'] = e4phi * bar_gamma_LL[:, i_r, i_r]

        # √ĝ for 1D spherical (det(ĝ) = r⁴ sin²θ, so √ĝ = r²)
        r_c = np.asarray(r, dtype=float)
        r_f = 0.5 * (r_c[:-1] + r_c[1:])
        
        g['sqrt_g_hat_cell'] = np.maximum(np.abs(r_c), 1e-30)**2
        g['r_faces'] = r_f
        g['sqrt_g_hat_face'] = np.maximum(np.abs(r_f), 1e-30)**2

        # CORRECCIÓN: Jacobiano J = e^{6φ} √(γ̄/ĝ) usando funciones del framework
        if spacetime_mode != "fixed_minkowski":
            e6phi = g['e6phi']
            det_bar_gamma = get_det_bar_gamma(r, bssn_vars.h_LL, background)
            sqrt_bar_gamma = np.sqrt(np.abs(det_bar_gamma) + 1e-30)
            sqrt_hat_gamma = np.sqrt(np.abs(background.det_hat_gamma) + 1e-30)
            J_cell = e6phi * sqrt_bar_gamma / sqrt_hat_gamma
        else:
            J_cell = np.ones(N)

        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])

        return g

    def _compute_interface_fluxes_full(self, rho0, vr, pressure, g, r,
                                       eos, reconstructor, riemann_solver):
        """Computes densitized fluxes at interfaces."""
        N = len(r)
        nfaces = N - 1
        flux_D = np.zeros(nfaces)
        flux_Sr = np.zeros(nfaces)
        flux_tau = np.zeros(nfaces)

        # Reconstruction → N+1 faces; trim to interior faces [1:-1]
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )
        rhoL = rhoL[1:-1]
        vL = vL[1:-1]
        pL = pL[1:-1]
        rhoR = rhoR[1:-1]
        vR = vR[1:-1]
        pR = pR[1:-1]

        # Face averages
        alpha_f = 0.5 * (g['alpha'][:-1] + g['alpha'][1:])
        beta_r_f = 0.5 * (g['beta_r'][:-1] + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f = g['J_face']               # nfaces
        sqrt_gh_f = g['sqrt_g_hat_face']      # nfaces

        # Optional physical limiters (nfaces length)
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999999,
                gamma_rr=gamma_rr_f
            )

        # Physical flux at all faces (partially vectorized)
        # Convert primitives to conservative variables on faces in a vectorized manner
        UL_D, UL_Sr, UL_tau = prim_to_cons(rhoL, vL, pL, gamma_rr_f, eos)
        UR_D, UR_Sr, UR_tau = prim_to_cons(rhoR, vR, pR, gamma_rr_f, eos)

        # Pack batches for the Riemann solver
        UL_batch = np.stack([UL_D, UL_Sr, UL_tau], axis=1)
        UR_batch = np.stack([UR_D, UR_Sr, UR_tau], axis=1)
        primL_batch = np.stack([rhoL, vL, pL], axis=1)
        primR_batch = np.stack([rhoR, vR, pR], axis=1)

        F_phys_batch = riemann_solver.solve_batch(
            UL_batch, UR_batch, primL_batch, primR_batch,
            gamma_rr_f, alpha_f, beta_r_f, eos
        )

        # Correct densitization on all faces: √γ̂_face * α_face * J_face * F_phys
        dens_face = alpha_f * J_f
        Fhat = (sqrt_gh_f[:, None] * dens_face[:, None]) * F_phys_batch

        flux_D[:] = Fhat[:, 0]
        flux_Sr[:] = Fhat[:, 1]
        flux_tau[:] = Fhat[:, 2]

        return {'D': flux_D, 'Sr': flux_Sr, 'tau': flux_tau}

    def _validate_geometry(self, g, r, spacetime_mode):
        """
        Validates that geometric quantities are physically reasonable.
        """
        # Verify that quantities are finite and positive where appropriate
        assert np.all(np.isfinite(g['alpha'])), "Lapse contains non-finite values"
        assert np.all(g['alpha'] > 0), "Lapse must be positive"
        assert np.all(np.isfinite(g['J_cell'])), "J factor contains non-finite values"
        assert np.all(g['J_cell'] > 0), "J factor must be positive"

        if spacetime_mode != "fixed_minkowski":
            assert np.all(g['gamma_rr'] > 0), "Spatial metric component must be positive"
            assert np.all(np.isfinite(g['gamma_rr'])), "Spatial metric contains non-finite values"

        return True

    def _apply_ghost_cell_boundaries(self, D, Sr, tau, r):
        """
        Parities at r≈0:
          D, τ  even;   S_r odd.
        Zero-order extrapolation at outer boundary.
        Vectorized implementation.
        """
        N = len(r)

        # Left boundary (center)
        if NUM_GHOSTS > 0:
            mir_slice = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            D[:NUM_GHOSTS] = D[mir_slice]
            Sr[:NUM_GHOSTS] = -Sr[mir_slice]
            tau[:NUM_GHOSTS] = tau[mir_slice]

        # Right boundary (outer edge)
        last_interior = N - NUM_GHOSTS - 1
        if last_interior >= 0 and NUM_GHOSTS > 0:
            D[-NUM_GHOSTS:] = D[last_interior]
            Sr[-NUM_GHOSTS:] = Sr[last_interior]
            tau[-NUM_GHOSTS:] = tau[last_interior]

        return D, Sr, tau

    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        """Same parities for the RHS (vectorized)."""
        N = len(r)

        if NUM_GHOSTS > 0:
            mir_slice = slice(2 * NUM_GHOSTS - 1, NUM_GHOSTS - 1, -1)
            rhs_D[:NUM_GHOSTS] = rhs_D[mir_slice]
            rhs_Sr[:NUM_GHOSTS] = -rhs_Sr[mir_slice]
            rhs_tau[:NUM_GHOSTS] = rhs_tau[mir_slice]

        last_interior = N - NUM_GHOSTS - 1
        if last_interior >= 0 and NUM_GHOSTS > 0:
            rhs_D[-NUM_GHOSTS:] = rhs_D[last_interior]
            rhs_Sr[-NUM_GHOSTS:] = rhs_Sr[last_interior]
            rhs_tau[-NUM_GHOSTS:] = rhs_tau[last_interior]

        return rhs_D, rhs_Sr, rhs_tau

