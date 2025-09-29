# valencia_reference_metric.py (FULL reference-metric approach, 1D spherical)

import numpy as np

from source.bssn.tensoralgebra import get_bar_gamma_LL, get_det_bar_gamma, get_bar_A_LL
from source.backgrounds.sphericalbackground import i_r
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons
# Avoid circular import - import dynamically when needed


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
    """

    def compute_rhs2(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        ∂_t(J U) + 𝔻̂_r[(f)^r] = J S_phys, con la expansión numérica:
        - D, τ : 𝔻̂_r v^r = (1/√ĝ) ∂_r (√ĝ v^r)
        - S_r : 𝔻̂_r (f_S)^r{}_r = (1/√ĝ) ∂_r (√ĝ (f_S)^r{}_r) − (f_S)^k{}_j 𝛤̂^j{}_{rk}
        En 1D esférico y simetría esférica, el término final reduce a + α J p · 𝛤̂^k{}_{rk} = + α J p · ∂_r ln√ĝ.
        """

        # Paso de malla
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            dr = float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            dr = float(grid.dr)
        else:
            dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

        # Conservadas + BC
        D  = D.copy(); Sr = Sr.copy(); tau = tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # Geometría y pesos de referencia
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)
        N = len(r)
        gh_cell = g['sqrt_g_hat_cell']   # √ĝ en centros
        gh_face = g['sqrt_g_hat_face']   # √ĝ en caras

        # Flujos F_phys en caras y densitización √ĝ·α·J
        flux_hat = self._compute_interface_fluxes_full(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # Divergencias básicas (vectoriales) para D y τ
        rhs_D   = np.zeros(N)
        rhs_Sr  = np.zeros(N)
        rhs_tau = np.zeros(N)

        div_D   = -(np.diff(flux_hat['D'  ])) / (gh_cell[1:-1] * dr + 1e-30)
        div_tau = -(np.diff(flux_hat['tau'])) / (gh_cell[1:-1] * dr + 1e-30)

        rhs_D [1:-1] = div_D
        rhs_tau[1:-1]= div_tau

        # Momento: (1/√ĝ)∂_r(√ĝ (f_S)^r{}_r)  +  corrección covariante de índice bajo  (ecuación (37))
        div_Sr = -(np.diff(flux_hat['Sr'])) / (gh_cell[1:-1] * dr + 1e-30)
        rhs_Sr[1:-1] = div_Sr

        # Γ̂^k_{rk} = ∂_r ln √ĝ  (discreto compatible con el flujo con √ĝ)
        Gamma_trace_r = (gh_face[1:] - gh_face[:-1]) / (gh_cell[1:-1] * dr + 1e-30)
        # Corrección: + α J p · Γ̂^k_{rk}  (anula 2p/r para p=cte y deja ∂_r p)
        rhs_Sr[1:-1] += (g['alpha'][1:-1] * g['J_cell'][1:-1]) * pressure[1:-1] * Gamma_trace_r

        # Fuentes físicas (geométricas) de (34) y (47) – sólo S_r y τ
        src_Sr, src_tau = self._compute_sources_full(
            pressure, g, bssn_vars, bssn_d1,
            rho0, h, W, vr, spacetime_mode, r, background
        )
        rhs_Sr  += src_Sr
        rhs_tau += src_tau

        # Pasar de ∂_t(JU) a ∂_tU
        J = g['J_cell'] + 1e-30
        rhs_D   = rhs_D   / J
        rhs_Sr  = rhs_Sr  / J
        rhs_tau = rhs_tau / J

        # BCs del RHS
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(rhs_D, rhs_Sr, rhs_tau, r)
        return rhs_D, rhs_Sr, rhs_tau


    def _compute_sources_full2(self, pressure, g, bssn_vars, bssn_d1,
                            rho0, h, W, vr, spacetime_mode, r, background):
        """
        Fuentes físicas para S_r y τ (devolvemos J·S_phys), consistentes con:
        - Momento: ecuación (34)
        - Energía: ecuación (47)
        Nota: en Minkowski las fuentes físicas son 0. El término 2p/r ya lo aporta
            la corrección covariante del flujo mixto (en compute_rhs), y se cancela
            con p=cte como debe.
        """
        N       = len(r)
        J_cell  = g['J_cell']
        alpha   = g['alpha']
        beta_u  = g['beta_r']        # β^r (contravariante)
        gamma_rr= g['gamma_rr']

        # --- Caso Minkowski: fuentes físicas nulas (evitamos doble conteo) ---
        if spacetime_mode == "fixed_minkowski":
            return J_cell * np.zeros(N), J_cell * np.zeros(N)

        # --- Componentes T^{μν} (fluido perfecto) ---
        T00, T0r, Trr = self.T_components_from_primitives(
            rho0=rho0, vr=vr, p=pressure, W=W, h=h,
            alpha=alpha, beta_r=beta_u, gamma_rr=gamma_rr
        )

        # --- Derivadas geométricas necesarias ---
        dalpha_dr = np.zeros(N)
        if getattr(bssn_d1, 'lapse', None) is not None:
            lapse_der = np.asarray(bssn_d1.lapse)
            if lapse_der.ndim >= 2 and lapse_der.shape[1] > i_r:
                dalpha_dr = lapse_der[:, i_r]

        # D̂_r β^r = ∂_r β^r en 1D esférico
        hat_D_beta_r = np.zeros(N)
        if getattr(bssn_d1, 'shift_U', None) is not None:
            shift_d1 = np.asarray(bssn_d1.shift_U)
            if shift_d1.ndim >= 3 and shift_d1.shape[1] > i_r and shift_d1.shape[2] > i_r:
                hat_D_beta_r = shift_d1[:, i_r, i_r]

        # D̂_r γ_{rr} = e^{4φ}[ ∂_r \barγ_{rr} + 4 \barγ_{rr} ∂_r φ ]  (ecuación (35))
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        e4phi   = np.exp(4.0 * phi_arr)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)   # \barγ_ij
        bar_gamma_rr = bar_gamma_LL[:, i_r, i_r]

        dphi_dr = np.zeros(N)
        if getattr(bssn_d1, 'phi', None) is not None:
            phi_der = np.asarray(bssn_d1.phi)
            if phi_der.ndim >= 2 and phi_der.shape[1] > i_r:
                dphi_dr = phi_der[:, i_r]

        dbar_gamma_rr_dr = np.zeros(N)
        if getattr(bssn_d1, 'h_LL', None) is not None:
            h_d1 = np.asarray(bssn_d1.h_LL)
            if h_d1.ndim >= 4 and h_d1.shape[1] > i_r and h_d1.shape[2] > i_r and h_d1.shape[3] > i_r:
                scaling_rr     = background.scaling_matrix[:, i_r, i_r]
                dscaling_rr_dr = background.d1_scaling_matrix[:, i_r, i_r, i_r]
                h_rr           = bssn_vars.h_LL[:, i_r, i_r]
                dh_rr_dr       = h_d1[:, i_r, i_r, i_r]
                dbar_gamma_rr_dr = scaling_rr * dh_rr_dr + h_rr * dscaling_rr_dr

        hat_D_gamma_rr = e4phi * (dbar_gamma_rr_dr + 4.0 * bar_gamma_rr * dphi_dr)

        # K_rr = A_rr + (1/3) γ_rr K
        K = getattr(bssn_vars, 'K', np.zeros(N))
        bar_A_LL      = get_bar_A_LL(r, bssn_vars, background)
        A_rr_physical = e4phi * bar_A_LL[:, i_r, i_r]
        K_rr          = A_rr_physical + (gamma_rr * K) / 3.0

        # Combinación estándar que aparece repetidamente
        combo = (T00 * (beta_u**2) + 2.0 * T0r * beta_u + Trr)

        # Términos fuente físicos (no densitizados), ecuacs. (34) y (47)
        src_Sr_phys  = alpha * ( - T00 * dalpha_dr
                                + (T00 * (gamma_rr*beta_u) + T0r * gamma_rr) * hat_D_beta_r
                                + 0.5 * combo * hat_D_gamma_rr )

        src_tau_phys = alpha * ( combo * K_rr
                                - (T00 * beta_u + T0r) * dalpha_dr )

        # Devolvemos J·S_phys (la división por J se hace en compute_rhs)
        return J_cell * src_Sr_phys, J_cell * src_tau_phys


    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        ∂_t(J U) + (1/√ĝ) ∂_r[ √ĝ (α J F^r_phys) ] = J S_phys,
        con la corrección de conexión que exige la expansión de la divergencia covariante
        para el tensor mixto de momento (1309.7808v2, ec. (37)).
        """

        # --- paso de malla ---
        if hasattr(grid, 'derivs') and hasattr(grid.derivs, 'dx'):
            dr = float(grid.derivs.dx)
        elif hasattr(grid, 'dr'):
            dr = float(grid.dr)
        else:
            dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

        # --- CONSERVADAS + BCs ---
        D  = D.copy()
        Sr = Sr.copy()
        tau = tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # --- Geometría ---
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)
        N = len(r)

        # --- Flujos (densitizados en caras con √ĝ · α · J) ---
        flux_hat = self._compute_interface_fluxes_full(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # --- Divergencia con peso √ĝ ---
        rhs_D   = np.zeros(N)
        rhs_Sr  = np.zeros(N)
        rhs_tau = np.zeros(N)

        gh_cell = g['sqrt_g_hat_cell']     # √ĝ en centros (r^2)
        gh_face = g['sqrt_g_hat_face']     # √ĝ en caras

        div_D   = -(np.diff(flux_hat['D'  ])) / (gh_cell[1:-1] * dr + 1e-30)
        div_Sr  = -(np.diff(flux_hat['Sr' ])) / (gh_cell[1:-1] * dr + 1e-30)
        div_tau = -(np.diff(flux_hat['tau'])) / (gh_cell[1:-1] * dr + 1e-30)

        rhs_D [1:-1] = div_D
        rhs_Sr[1:-1] = div_Sr
        rhs_tau[1:-1]= div_tau

        det_hat = np.asarray(background.det_hat_gamma, dtype=float)                # shape (N,)
        ddet_dr = np.asarray(background.d1_det_hat_gamma, dtype=float)[:, i_r]     # ∂r det(γ̂)
        Gamma_trace_r = 0.5 * ddet_dr / (det_hat + 1e-30)                          # = ∂r ln √ĝ
        #else:
            # Fallback compatible con el flujo densitizado: ( √ĝ_{i+1/2} − √ĝ_{i−1/2} ) / ( √ĝ_i Δr )
         #   Gamma_trace_r = (gh_face[1:] - gh_face[:-1]) / (gh_cell[1:-1] * dr + 1e-30)

        # Añade + (α J p) Γ̂^k_{rk} sólo al RHS de S_r (tensor mixto)
        rhs_Sr[1:-1] += (g['alpha'][1:-1] * g['J_cell'][1:-1]) * pressure[1:-1] * Gamma_trace_r[1:-1]

        # --- Fuentes físicas (J·S_phys): sólo afectan S_r y τ ---
        src_Sr, src_tau = self._compute_sources_full(
            pressure, g, bssn_vars, bssn_d1,
            rho0, h, W, vr, spacetime_mode, r, background
        )
        rhs_Sr  += src_Sr
        rhs_tau += src_tau

        # --- Pasar de ∂_t(JU) a ∂_t U ---
        J = g['J_cell'] + 1e-30
        rhs_D   = rhs_D   / J
        rhs_Sr  = rhs_Sr  / J
        rhs_tau = rhs_tau / J

        # --- BCs del RHS ---
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(rhs_D, rhs_Sr, rhs_tau, r)
        return rhs_D, rhs_Sr, rhs_tau


    def _compute_sources_full(self, pressure, g, bssn_vars, bssn_d1,
                            rho0, h, W, vr, spacetime_mode, r, background):
        """
        Fuentes físicas para S_r y τ (devolvemos J·S_phys).
        - Minkowski: 0 (las únicas correcciones geométricas al momento ya se añaden en compute_rhs).
        - Dinámico: implementación de (s_S)_r y s_τ según 1309.7808v2 (ecs. (34) y (47)),
        con  \hat D_r γ_{rr} = e^{4φ} [ 4 \barγ_{rr} ∂_r φ + (\hat D \barγ)_{rrr} ].
        """
        import numpy as np
        from source.backgrounds.sphericalbackground import i_r
        # Usamos la utilidad que ya implementa \hat D \barγ conforme a Baumgarte (ec. 25).  :contentReference[oaicite:4]{index=4}
        from source.bssn.tensoralgebra import get_hat_D_bar_gamma_LL

        N       = len(r)
        J_cell  = g['J_cell']
        alpha   = g['alpha']
        beta_u  = g['beta_r']          # β^r (contravariante)
        gamma_rr= g['gamma_rr']

        # --- Minkowski: sin fuentes físicas ---
        if spacetime_mode == "fixed_minkowski":
            return J_cell * np.zeros(N), J_cell * np.zeros(N)

        # --- Componentes T^{μν} desde primitivas (mixto con índice arriba) ---
        T00, T0u, Tuu = self.T_components_from_primitives(
            rho0=rho0, vr=vr, p=pressure, W=W, h=h,
            alpha=alpha, beta_r=beta_u, gamma_rr=gamma_rr
        )

        # --- Derivadas de calibre ---
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
        # \hat D_r β^r = ∂_r β^r + \hatΓ^r_{r r} β^r = ∂_r β^r (en fondo esférico)  :contentReference[oaicite:5]{index=5}
        hat_D_beta_r = dbeta_dr

        # --- \hat D_r γ_{rr} = e^{4φ} [ 4 \barγ_{rr} ∂_r φ + (\hat D \barγ)_{rrr} ]  (ec. (35))
        phi_arr = np.asarray(bssn_vars.phi, dtype=float)
        e4phi   = np.exp(4.0 * phi_arr)
        from source.bssn.tensoralgebra import get_bar_gamma_LL
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        bar_gamma_rr = bar_gamma_LL[:, i_r, i_r]

        dphi_dr = np.zeros(N)
        if getattr(bssn_d1, 'phi', None) is not None:
            phi_der = np.asarray(bssn_d1.phi)
            if phi_der.ndim >= 2 and phi_der.shape[1] > i_r:
                dphi_dr = phi_der[:, i_r]

        hat_D_bar_gamma = get_hat_D_bar_gamma_LL(r, bssn_vars.h_LL, bssn_d1.h_LL, background)  # (N,3,3,3)
        hat_D_bar_gamma_rrr = hat_D_bar_gamma[:, i_r, i_r, i_r]
        hat_D_gamma_rr = e4phi * (4.0 * bar_gamma_rr * dphi_dr + hat_D_bar_gamma_rrr)  # :contentReference[oaicite:6]{index=6}

        # --- K_rr = A_rr + (1/3) γ_rr K ---
        K = getattr(bssn_vars, 'K', np.zeros(N))
        from source.bssn.tensoralgebra import get_bar_A_LL
        bar_A_LL      = get_bar_A_LL(r, bssn_vars, background)
        A_rr_physical = e4phi * bar_A_LL[:, i_r, i_r]
        K_rr          = A_rr_physical + (gamma_rr * K) / 3.0

        # --- Combinaciones estándar ---
        combo = (T00 * (beta_u**2) + 2.0 * T0u * beta_u + Tuu)

        # T^{0}{}_r = T^{00} β_r + T^{0r} γ_{rr}, con β_r = γ_{rr} β^r
        beta_l       = gamma_rr * beta_u
        T0_lower_r   = T00 * beta_l + T0u * gamma_rr

        # (s_S)_r  = α[ − T00 ∂_r α + T^{0}{}_r \hat D_r β^r + ½ combo · \hat D_r γ_{rr} ]
        src_Sr_phys  = alpha * ( - T00 * dalpha_dr
                                + T0_lower_r * hat_D_beta_r
                                + 0.5 * combo * hat_D_gamma_rr )

        # s_τ = α[ combo·K_rr − (T^{00} β^r + T^{0r}) ∂_r α ]
        src_tau_phys = alpha * ( combo * K_rr
                                - (T00 * beta_u + T0u) * dalpha_dr )

        # Densitizamos (J·S_phys). La división por J se hace en compute_rhs.
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

    @staticmethod
    def T_components_from_primitives(rho0, vr, p, W, h, alpha, beta_r, gamma_rr):
        """
        Devuelve (T^{00}, T^{0r}, T^{rr}) para un fluido perfecto en 3+1 (1D radial):
            T^{μν} = ρ0 h u^μ u^ν + p g^{μν}

        Convenciones:
        - u^0 = W / α
        - u^r = W ( v^r - β^r / α ), con β^r = γ^{rr} β_r
        - g^{00} = -1/α^2
        - g^{0r} = β^r/α^2
        - g^{rr} = γ^{rr} - (β^r)^2 / α^2

        Todos los argumentos pueden ser escalares o arrays de la misma longitud.
        """

        rho0  = np.asarray(rho0, dtype=float)
        vr    = np.asarray(vr,   dtype=float)
        p     = np.asarray(p,    dtype=float)
        W     = np.asarray(W,    dtype=float)
        h     = np.asarray(h,    dtype=float)
        alpha = np.asarray(alpha,dtype=float)
        beta_r= np.asarray(beta_r,dtype=float)
        grr   = 1.0/np.asarray(gamma_rr, dtype=float)  # γ^{rr}

        beta_u = grr * beta_r                  # β^r
        ut = W / (alpha + 1e-30)
        ur = W * (vr - beta_u/(alpha + 1e-30))

        g00 = -1.0/(alpha**2)
        g0r =  beta_u/(alpha**2)
        grr_eff = grr - (beta_u**2)/(alpha**2)

        T00 = rho0*h*ut*ut + p*g00
        T0r = rho0*h*ut*ur + p*g0r
        Trr = rho0*h*ur*ur + p*grr_eff
        return T00, T0r, Trr

