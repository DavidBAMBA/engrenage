# valencia_reference_metric.py (FULL reference-metric approach, 1D esférico)

import numpy as np

from source.bssn.tensoralgebra import get_bar_gamma_LL
from source.backgrounds.sphericalbackground import i_r
from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.cons2prim import prim_to_cons


class ValenciaReferenceMetric:
    """
    Formulación de Valencia con MÉTRICA DE REFERENCIA (full approach) para 1D esférico.
    Basado en Montero et al. (2013) arXiv:1309.7808 (enfoque "full").

    Ecuación en forma conservativa con métrica de referencia (1D):
        ∂_t( J U ) + (1/√ĝ) ∂_r [ √ĝ ( α J F^r_phys ) ] = J S_phys,
    donde
        J   = e^{6φ} √( \barγ / ĝ ),
        √ĝ  = √det(ĝ)   (en 1D esférico:  √ĝ = r^2),
        F^r_phys  es el flujo *no densitizado* (con ṽ^r = v^r − β^r/α),
        S_phys     son las fuentes físicas (geom./grav.) sin densitizar.

    Nota: el solver de Riemann devuelve F^r_phys; aquí aplicamos α·J y el peso √ĝ.
    """

    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        ∂t(J U) + (1/√ĝ) ∂r[ √ĝ (α J F^r_phys) ] = J S_phys
        con √ĝ = r^2. Para el momento usamos la forma de tensor mixto
        (ver eqs. (25), (37)); los aportes angulares (hoop stress) entran
        vía las fuentes J·S_phys.
        """
        N = len(r)
        #dr = float(grid.derivs.dx)  # CORRECCIÓN: usar grid.derivs.dx que es escalar
        dr = float(grid.dr)

        # 1) BCs en CONSERVADAS (no pisar arrays originales)
        D = D.copy()
        Sr = Sr.copy()
        tau = tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # 2) Geometría (α, β_r, e^{6φ}, γ_rr, √ĝ en celda/cara, J en celda/cara)
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # 3) Flujos en caras:  flux_hat = √ĝ_face * (α_face * J_face * F_phys)
        flux_hat = self._compute_interface_fluxes_full(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # 4) Divergencia covariante para cada ecuación (vector en D y τ)
        rhs_D = np.zeros(N)
        rhs_Sr = np.zeros(N)
        rhs_tau = np.zeros(N)

        sgc = g['sqrt_g_hat_cell']   # √ĝ en centros (= |r|^2)
        for i in range(NUM_GHOSTS, N - NUM_GHOSTS):
            inv_vol = 1.0 / (sgc[i] * dr + 1e-30)
            rhs_D[i] = -(flux_hat['D'][i] - flux_hat['D'][i-1]) * inv_vol
            rhs_Sr[i] = -(flux_hat['Sr'][i] - flux_hat['Sr'][i-1]) * inv_vol
            rhs_tau[i] = -(flux_hat['tau'][i] - flux_hat['tau'][i-1]) * inv_vol

        # 5) Fuentes J·S_phys
        src_Sr, src_tau = self._compute_sources_full(
            pressure, g, bssn_vars, bssn_d1, rho0, h, W, vr, spacetime_mode, r
        )
        rhs_Sr += src_Sr
        rhs_tau += src_tau

        # 6) BCs al RHS (paridades)
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(rhs_D, rhs_Sr, rhs_tau, r)
        return rhs_D, rhs_Sr, rhs_tau

    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        """Extrae cantidades geométricas con verificación robusta de dimensiones."""
        N = len(r)
        g = {}

        # Campos geométricos
        if spacetime_mode == "fixed_minkowski":
            g['alpha'] = np.ones(N)
            g['beta_r'] = np.zeros(N)
            g['e6phi'] = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            g['alpha'] = np.asarray(bssn_vars.lapse, dtype=float)
            
            # Shift con verificación de dimensiones
            if hasattr(bssn_vars, 'shift_U') and bssn_vars.shift_U is not None:
                shift_array = np.asarray(bssn_vars.shift_U)
                if shift_array.ndim >= 2 and shift_array.shape[1] > i_r:
                    g['beta_r'] = shift_array[:, i_r].astype(float)
                else:
                    g['beta_r'] = np.zeros(N)
            else:
                g['beta_r'] = np.zeros(N)
            
            # Factor conformal
            g['e6phi'] = np.exp(6.0 * np.asarray(bssn_vars.phi, dtype=float))
            
            # Métrica conforme
            bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            g['gamma_rr'] = g['e6phi'] * bar_gamma_LL[:, i_r, i_r]

        # √ĝ en centros / caras (en 1D esf.: √ĝ = r^2)
        r_c = np.asarray(r, dtype=float)
        r_f = 0.5 * (r_c[:-1] + r_c[1:])
        r_f_abs = np.maximum(np.abs(r_f), 1e-30)

        g['sqrt_g_hat_cell'] = np.maximum(np.abs(r_c), 1e-30)**2
        g['r_faces'] = r_f
        g['sqrt_g_hat_face'] = r_f_abs**2

        # Densitización J ≈ e^{6φ} (cuando γ̄ y ĝ comparten det en 1D)
        J_cell = g['e6phi']
        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])

        return g

    def _compute_interface_fluxes_full(self, rho0, vr, pressure, g, r,
                                       eos, reconstructor, riemann_solver):
        """Calcula flujos densitizados en interfaces."""
        N = len(r)
        nfaces = N - 1
        flux_D = np.zeros(nfaces)
        flux_Sr = np.zeros(nfaces)
        flux_tau = np.zeros(nfaces)

        # Reconstrucción → N+1 caras; recorte a caras interiores [1:-1]
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )
        rhoL = rhoL[1:-1]
        vL = vL[1:-1]
        pL = pL[1:-1]
        rhoR = rhoR[1:-1]
        vR = vR[1:-1]
        pR = pR[1:-1]

        # Promedios en cara
        alpha_f = 0.5 * (g['alpha'][:-1] + g['alpha'][1:])
        beta_r_f = 0.5 * (g['beta_r'][:-1] + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f = g['J_face']               # nfaces
        sqrt_gh_f = g['sqrt_g_hat_face']      # nfaces

        # Limitadores físicos opcionales (longitudes nfaces)
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999999,
                gamma_rr=gamma_rr_f
            )

        # Flujo físico en cada cara y densitización
        for j in range(nfaces):
            UL = prim_to_cons(rhoL[j], vL[j], pL[j], gamma_rr_f[j], eos)
            UR = prim_to_cons(rhoR[j], vR[j], pR[j], gamma_rr_f[j], eos)

            F_phys = riemann_solver.solve(
                UL, UR,
                (rhoL[j], vL[j], pL[j]),
                (rhoR[j], vR[j], pR[j]),
                gamma_rr_f[j], alpha_f[j], beta_r_f[j], eos
            )
            dens_face = alpha_f[j] * J_f[j]
            Fhat = sqrt_gh_f[j] * dens_face * F_phys  # √ĝ_face * (α_face J_face) * F_phys

            flux_D[j], flux_Sr[j], flux_tau[j] = Fhat[0], Fhat[1], Fhat[2]

        return {'D': flux_D, 'Sr': flux_Sr, 'tau': flux_tau}

    def _compute_sources_full(self, pressure, g, bssn_vars, bssn_d1,
                              rho0, h, W, vr, spacetime_mode, r):
        """
        Implementa fuentes físicas densitizadas con J. 
        En Minkowski fijo solo hoop stress +2p/r en Sr.
        En métrica dinámica incluye términos de acoplamiento gravitacional.
        """
        N = len(r)
        J_cell = g['J_cell']

        # --- caso Minkowski fijo: solo hoop stress en la ecuación de momento
        if spacetime_mode == "fixed_minkowski":
            r_safe = np.maximum(np.abs(np.asarray(r, dtype=float)), 1e-30)
            src_Sr_phys = 2.0 * pressure / r_safe   # Γ̂^θ_{rθ} = Γ̂^φ_{rφ} = 1/r  ⇒  +2p/r
            src_tau_phys = np.zeros(N)               # no hay fuentes geométricas en τ con Cowling
            return J_cell * src_Sr_phys, J_cell * src_tau_phys

        # --- caso métrica dinámica: términos completos de acoplamiento
        alpha = g['alpha']
        gamma_rr = g['gamma_rr']

        # T^{μν} para fluido perfecto
        T00 = rho0 * h * W**2 - pressure
        T0r = rho0 * h * W**2 * vr
        Trr = rho0 * h * W**2 * vr**2 + pressure

        # Derivadas BSSN (radial) - con verificación robusta de dimensiones
        dalpha_dr = np.zeros(N)
        dbeta_dr = np.zeros(N)
        dphi_dr = np.zeros(N)
        
        # Derivada del lapse
        if hasattr(bssn_d1, 'lapse') and bssn_d1.lapse is not None:
            lapse_deriv = np.asarray(bssn_d1.lapse)
            if lapse_deriv.ndim >= 2 and lapse_deriv.shape[1] > i_r:
                dalpha_dr = lapse_deriv[:, i_r]
        
        # Derivada del shift
        if hasattr(bssn_d1, 'shift_U') and bssn_d1.shift_U is not None:
            shift_deriv = np.asarray(bssn_d1.shift_U)
            if shift_deriv.ndim >= 3 and shift_deriv.shape[1] > i_r and shift_deriv.shape[2] > i_r:
                dbeta_dr = shift_deriv[:, i_r, i_r]  # ∂r β^r
        
        # Derivada del factor conformal
        if hasattr(bssn_d1, 'phi') and bssn_d1.phi is not None:
            phi_deriv = np.asarray(bssn_d1.phi)
            if phi_deriv.ndim >= 2 and phi_deriv.shape[1] > i_r:
                dphi_dr = phi_deriv[:, i_r]

        # Aproximación 1D para ∂r γ_rr a partir de φ (γ_rr = e^{4φ} \barγ_rr, con \barγ_rr≈1)
        d_gamma_rr_dr = gamma_rr * (4.0 * dphi_dr)

        # Hoop stress siempre presente
        r_safe = np.maximum(np.abs(np.asarray(r, dtype=float)), 1e-30)
        hoop_stress = 2.0 * pressure / r_safe

        # (sS)^r (ec. (34)), escrito en términos de componentes útiles en 1D
        src_Sr_phys = hoop_stress + (- T00 * dalpha_dr) + (T0r * dbeta_dr) + 0.5 * Trr * d_gamma_rr_dr

        # s_τ (ec. (47)) – usa K y gradientes de α
        K = getattr(bssn_vars, 'K', np.zeros(N))
        src_tau_phys = alpha * K * (pressure - T00)

        return J_cell * src_Sr_phys, J_cell * src_tau_phys

    def _apply_ghost_cell_boundaries(self, D, Sr, tau, r):
        """
        Paridades en r≈0:
          D, τ  pares;   S_r impar.
        Extrapolación de orden cero en la frontera externa.
        """
        N = len(r)

        # r=0 (lado izquierdo, NUM_GHOSTS celdas)
        for i in range(NUM_GHOSTS):
            mirror = 2 * NUM_GHOSTS - 1 - i
            D[i] = D[mirror]     # par
            Sr[i] = -Sr[mirror]    # impar
            tau[i] = tau[mirror]   # par

        # frontera externa (lado derecho)
        last_interior = N - NUM_GHOSTS - 1
        for k in range(1, NUM_GHOSTS + 1):
            idx = last_interior + k
            D[idx] = D[last_interior]
            Sr[idx] = Sr[last_interior]
            tau[idx] = tau[last_interior]
        return D, Sr, tau

    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        """Las mismas paridades para el RHS."""
        N = len(r)

        # r=0
        for i in range(NUM_GHOSTS):
            mirror = 2 * NUM_GHOSTS - 1 - i
            rhs_D[i] = rhs_D[mirror]     # par
            rhs_Sr[i] = -rhs_Sr[mirror]    # impar
            rhs_tau[i] = rhs_tau[mirror]   # par

        # frontera externa
        last_interior = N - NUM_GHOSTS - 1
        for k in range(1, NUM_GHOSTS + 1):
            idx = last_interior + k
            rhs_D[idx] = rhs_D[last_interior]
            rhs_Sr[idx] = rhs_Sr[last_interior]
            rhs_tau[idx] = rhs_tau[last_interior]
        return rhs_D, rhs_Sr, rhs_tau