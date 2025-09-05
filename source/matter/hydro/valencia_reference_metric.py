# valencia_reference_metric.py (FULL reference-metric approach, 1D esférico)

import numpy as np
from ...bssn.tensoralgebra import get_bar_gamma_LL
from ...backgrounds.sphericalbackground import i_r
from ...core.spacing import NUM_GHOSTS
from .cons2prim import prim_to_cons  # función centralizada para U=(D,S_r,τ)



class ValenciaReferenceMetric:
    """
    Formulación de Valencia con MÉTRICA DE REFERENCIA (full approach) para 1D esférico.
    Basado en Montero et al. (2013) arXiv:1309.7808 (enfoque "full").

    Ecuación en forma conservativa con métrica de referencia (1D):
        ∂_t( J U ) + (1/√ĝ) ∂_r [ √ĝ ( α J F^r_phys ) ] = J S_phys,
    donde
        J   = e^{6φ} √( \barγ / ĝ ),
        √ĝ  = √det(ĝ)   (en 1D esférico:  √ĝ = r^2),
        F^r_phys  es el flujo *no densitizado* (con ṽ^r = v^r − β^r/α),
        S_phys     son las fuentes físicas (geom./grav.) sin densitizar.

    Nota: el solver de Riemann devuelve F^r_phys; aquí aplicamos α·J y el peso √ĝ.
    """

    # ------------------------------------------------------------------
    # PÚBLICO: RHS con enfoque FULL reference-metric
    # ------------------------------------------------------------------
    def compute_rhs2(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        Calcula RHS para (D, S_r, τ) siguiendo el enfoque *full* de métrica de referencia.
        * Aplica 3 ghost layers (NUM_GHOSTS).
        * Usa divergencia covariante: (1/√ĝ) ∂_r [ √ĝ (α·J·F^r_phys) ].
        * Agrega fuentes densitizadas como J·S_phys.

        Parámetros clave:
          - r: centros de celda (incluye fantasmas)
          - grid.dx: Δr (se asume uniforme)
          - riemann_solver: debe devolver F^r_phys (no densitizado)
        """
        N = len(r)
        dr = grid.dx

        # 1) BCs en las CONSERVADAS (3 fantasmas)
        D, Sr, tau = self._apply_ghost_cell_boundaries(D.copy(), Sr.copy(), tau.copy(), r)

        # 2) Geometría
        geom = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # 3) Flujos en interfaces (α·J·F^r_phys) ponderados con √ĝ en la cara
        flux_hat = self._compute_interface_fluxes_full(
            rho0, vr, pressure, geom, r, eos, reconstructor, riemann_solver
        )
        # flux_hat: diccionario con 'D','Sr','tau' que almacenan
        #           √ĝ_face * (α_face * J_face * F^r_phys) en cada interfaz

        # 4) Divergencia covariante (celdas interiores):
        rhs_D   = np.zeros(N)
        rhs_Sr  = np.zeros(N)
        rhs_tau = np.zeros(N)

        sqrt_g_hat_cell = geom['sqrt_g_hat_cell']  # r^2 en centros

        for i in range(NUM_GHOSTS, N - NUM_GHOSTS):
            inv_vol = 1.0 / (sqrt_g_hat_cell[i] * dr + 1e-30)
            rhs_D[i]   = - (flux_hat['D'][i]   - flux_hat['D'][i-1])   * inv_vol
            rhs_Sr[i]  = - (flux_hat['Sr'][i]  - flux_hat['Sr'][i-1])  * inv_vol
            rhs_tau[i] = - (flux_hat['tau'][i] - flux_hat['tau'][i-1]) * inv_vol


        # 5) Fuentes físicas → densitizadas como J·S_phys (NO añadir manualmente ±(2/r)F;
        #    el operador con √ĝ ya incorpora los términos de conexión de continuidad/energía).
        src_Sr, src_tau = self._compute_sources_full(
            pressure, geom, bssn_vars, bssn_d1, rho0, h, W, vr, spacetime_mode, r
        )
        rhs_Sr  += src_Sr
        rhs_tau += src_tau

        # 6) BCs al RHS
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(rhs_D, rhs_Sr, rhs_tau, r)
        return rhs_D, rhs_Sr, rhs_tau

    # ------------------------------------------------------------------
    # RHS (FULL reference-metric, 1D esférico)
    # ------------------------------------------------------------------
    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                    r, bssn_vars, bssn_d1, background, spacetime_mode,
                    eos, grid, reconstructor, riemann_solver):
        """
        ∂t(J U) + (1/√ĝ) ∂r[ √ĝ (α J F^r_phys) ] = J S_phys
        con √ĝ = r^2. Para el momento usamos la forma de tensor mixto
        (ver eqs. (25), (37)); los aportes angulares (hoop stress) entran
        vía las fuentes J·S_phys.
        """
        N  = len(r)
        dr = float(grid.dx)

        # 1) BCs en CONSERVADAS (no pisar arrays originales)
        D   = D.copy();  Sr  = Sr.copy();  tau = tau.copy()
        D, Sr, tau = self._apply_ghost_cell_boundaries(D, Sr, tau, r)

        # 2) Geometría (α, β_r, e^{6φ}, γ_rr, √ĝ en celda/cara, J en celda/cara)
        g = self._extract_geometry(r, bssn_vars, spacetime_mode, background)

        # 3) Flujos en caras:  flux_hat = √ĝ_face * (α_face * J_face * F_phys)
        flux_hat = self._compute_interface_fluxes_full(
            rho0, vr, pressure, g, r, eos, reconstructor, riemann_solver
        )

        # 4) Divergencia covariante para cada ecuación (vector en D y τ)
        rhs_D   = np.zeros(N)
        rhs_Sr  = np.zeros(N)
        rhs_tau = np.zeros(N)

        sgc = g['sqrt_g_hat_cell']   # √ĝ en centros (= |r|^2)
        for i in range(NUM_GHOSTS, N - NUM_GHOSTS):
            inv_vol = 1.0 / (sgc[i] * dr + 1e-30)
            rhs_D[i]   = - (flux_hat['D'][i]   - flux_hat['D'][i-1])   * inv_vol
            rhs_Sr[i]  = - (flux_hat['Sr'][i]  - flux_hat['Sr'][i-1])  * inv_vol
            rhs_tau[i] = - (flux_hat['tau'][i] - flux_hat['tau'][i-1]) * inv_vol

        # 5) Fuentes J·S_phys
        src_Sr, src_tau = self._compute_sources_full(
            pressure, g, bssn_vars, bssn_d1, rho0, h, W, vr, spacetime_mode, r
        )
        rhs_Sr  += src_Sr
        rhs_tau += src_tau

        # 6) BCs al RHS (paridades)
        rhs_D, rhs_Sr, rhs_tau = self._apply_rhs_boundary_conditions(rhs_D, rhs_Sr, rhs_tau, r)
        return rhs_D, rhs_Sr, rhs_tau

    # ------------------------------------------------------------------
    # GEOMETRÍA Y FACTORES
    # ------------------------------------------------------------------
    def _extract_geometry2(self, r, bssn_vars, spacetime_mode, background):
        """
        Construye diccionario geométrico con:
          alpha, beta_r, e6phi, gamma_rr,
          J_cell             = e^{6φ} √( \barγ / ĝ ) (en 1D: ≈ e^{6φ}),
          sqrt_g_hat_cell    = √ĝ en *centros*  (= r^2),
          sqrt_g_hat_face    = √ĝ en *caras*    (= r_face^2),
          promedios en cara: alpha_f, beta_r_f, gamma_rr_f, J_face.
        """
        N = len(r)
        g = {}

        # Seguridad numérica
        r_safe_c = np.maximum(np.asarray(r, dtype=float), 1e-30)

        if spacetime_mode == "fixed_minkowski":
            g['alpha']    = np.ones(N)
            g['beta_r']   = np.zeros(N)
            g['e6phi']    = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            g['alpha']    = np.asarray(bssn_vars.lapse, dtype=float)
            g['beta_r']   = np.asarray(bssn_vars.shift_U[:, i_r, i_r], dtype=float)
            g['e6phi']    = np.exp(6.0 * np.asarray(bssn_vars.phi, dtype=float))
            # γ_rr total a partir de la métrica conforme
            bar_gamma_LL  = get_bar_gamma_LL(bssn_vars)
            g['gamma_rr'] = g['e6phi'] * bar_gamma_LL[:, i_r, i_r]

        # En 1D esférico con referencia: √ĝ = r^2
        sqrt_g_hat_cell = r_safe_c**2
        g['sqrt_g_hat_cell'] = sqrt_g_hat_cell

        # Caras (malla uniforme o no): r_face[i] = promedio entre r[i-1] y r[i]
        r_faces = 0.5 * (r_safe_c[:-1] + r_safe_c[1:])
        r_safe_f = np.maximum(r_faces, 1e-30)
        g['r_faces'] = r_faces
        g['sqrt_g_hat_face'] = r_safe_f**2

        # Enfoque full: J = e^{6φ} √(\barγ/ĝ). En 1D con det(\barγ)=det(ĝ) ⇒ J ≈ e^{6φ}.
        # Si necesitas las razones exactas, reemplaza estas dos líneas por
        # cálculos explícitos de √\barγ y √ĝ.
        J_cell = g['e6phi']  # ~ e^{6φ}
        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])

        return g


    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        N = len(r)
        g = {}

        # Campos geométricos
        if spacetime_mode == "fixed_minkowski":
            g['alpha']    = np.ones(N)
            g['beta_r']   = np.zeros(N)
            g['e6phi']    = np.ones(N)
            g['gamma_rr'] = np.ones(N)
        else:
            g['alpha']    = np.asarray(bssn_vars.lapse, dtype=float)
            g['beta_r']   = np.asarray(bssn_vars.shift_U[:, i_r, i_r], dtype=float)
            g['e6phi']    = np.exp(6.0 * np.asarray(bssn_vars.phi, dtype=float))
            bar_gamma_LL  = get_bar_gamma_LL(bssn_vars)
            g['gamma_rr'] = g['e6phi'] * bar_gamma_LL[:, i_r, i_r]

        # √ĝ en centros / caras (en 1D esf.: √ĝ = r^2)
        r_c      = np.asarray(r, dtype=float)
        r_f      = 0.5 * (r_c[:-1] + r_c[1:])
        r_f_abs  = np.maximum(np.abs(r_f), 1e-30)

        g['sqrt_g_hat_cell'] = np.maximum(np.abs(r_c), 1e-30)**2
        g['r_faces']         = r_f
        g['sqrt_g_hat_face'] = r_f_abs**2

        # Densitización J ≈ e^{6φ} (cuando γ̄ y ĝ comparten det en 1D)
        J_cell = g['e6phi']
        g['J_cell'] = J_cell
        g['J_face'] = 0.5 * (J_cell[:-1] + J_cell[1:])

        # *** volumen exacto de cascarón: ∫ r^2 dr = (r^3)/3 ***
        # usa los valores en caras (siempre positivo por el cubo del valor absoluto)
        r3_f = (r_f_abs**3)
        vol_hat = np.zeros(N)
        vol_hat[1:-1] = (r3_f[1:] - r3_f[:-1]) / 3.0   # válido en celdas interiores
        # bordes: copia vecinos (no se usan en el bucle interior pero evita NaNs)
        vol_hat[0]     = vol_hat[1]
        vol_hat[-1]    = vol_hat[-2]
        g['vol_hat_cell'] = vol_hat

        return g

    # ------------------------------------------------------------------
    # FLUJOS EN CARAS (FULL): √ĝ_face * ( α_face * J_face * F^r_phys )
    # ------------------------------------------------------------------
    def _compute_interface_fluxes_full2(self, rho0, vr, pressure, g, r,
                                    eos, reconstructor, riemann_solver):
        """
        Devuelve diccionario de arrays de longitud N-1 con:
            flux_hat['D'/'Sr'/'tau'][j] = √ĝ_face * ( α_face * J_face * F^r_phys )[j]
        para j = 0..nfaces-1 (caras interiores).
        """
        N = len(r)
        nfaces = N - 1

        flux_D_hat   = np.zeros(nfaces)
        flux_Sr_hat  = np.zeros(nfaces)
        flux_tau_hat = np.zeros(nfaces)

        # 1) Reconstrucción a caras (el reconstructor típicamente devuelve N+1 caras)
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )

        # >>>>>>> RECORTE a caras interiores: 1..N-1  (quita 0 y N)
        rhoL = rhoL[1:-1]; vL = vL[1:-1]; pL = pL[1:-1]
        rhoR = rhoR[1:-1]; vR = vR[1:-1]; pR = pR[1:-1]
        # Ahora todas estas arrays tienen longitud nfaces (= N-1)

        # 2) Promedios geométricos en caras (ya son nfaces)
        alpha_f    = 0.5 * (g['alpha'][:-1]    + g['alpha'][1:])
        beta_r_f   = 0.5 * (g['beta_r'][:-1]   + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f        = g['J_face']            # nfaces
        sqrt_gh_f  = g['sqrt_g_hat_face']   # nfaces

        # 3) Limitadores físicos con longitudes coherentes (todas nfaces)
        if hasattr(reconstructor, "apply_physical_limiters"):
            (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
                (rhoL, vL, pL), (rhoR, vR, pR),
                atmosphere_rho=1e-13, p_floor=1e-15, v_max=0.999999,
                gamma_rr=gamma_rr_f
            )

        # 4) Riemann HLLE/HLLC en cada cara interior (índice j directo)
        for j in range(nfaces):
            # Conservadas desde primitivos en la cara j
            UL = prim_to_cons(rhoL[j], vL[j], pL[j], gamma_rr_f[j], eos)
            UR = prim_to_cons(rhoR[j], vR[j], pR[j], gamma_rr_f[j], eos)

            # Flujo físico NO densitizado (usa ṽ^r = v^r − β^r/α)
            F_phys = riemann_solver.solve(
                UL, UR,
                (rhoL[j], vL[j], pL[j]),
                (rhoR[j], vR[j], pR[j]),
                gamma_rr_f[j], alpha_f[j], beta_r_f[j], eos
            )

            # Densitización y peso √ĝ en la cara:  √ĝ_f * (α_f * J_f) * F_phys
            dens_f = alpha_f[j] * J_f[j]
            Fhat   = sqrt_gh_f[j] * dens_f * F_phys

            flux_D_hat[j]   = Fhat[0]
            flux_Sr_hat[j]  = Fhat[1]
            flux_tau_hat[j] = Fhat[2]

        return {'D': flux_D_hat, 'Sr': flux_Sr_hat, 'tau': flux_tau_hat}


    def _compute_interface_fluxes_full(self, rho0, vr, pressure, g, r,
                                    eos, reconstructor, riemann_solver):
        N = len(r)
        nfaces = N - 1
        flux_D   = np.zeros(nfaces)
        flux_Sr  = np.zeros(nfaces)
        flux_tau = np.zeros(nfaces)

        # Reconstrucción → N+1 caras; recorte a caras interiores [1:-1]
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="reflecting"
        )
        rhoL = rhoL[1:-1];  vL = vL[1:-1];  pL = pL[1:-1]
        rhoR = rhoR[1:-1];  vR = vR[1:-1];  pR = pR[1:-1]

        # Promedios en cara
        alpha_f    = 0.5 * (g['alpha'][:-1]    + g['alpha'][1:])
        beta_r_f   = 0.5 * (g['beta_r'][:-1]   + g['beta_r'][1:])
        gamma_rr_f = 0.5 * (g['gamma_rr'][:-1] + g['gamma_rr'][1:])
        J_f        = g['J_face']               # nfaces
        sqrt_gh_f  = g['sqrt_g_hat_face']      # nfaces

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

    # ------------------------------------------------------------------
    # FUENTES (FULL):  J · S_phys
    # ------------------------------------------------------------------
    def _compute_sources_full2(self, pressure, g, bssn_vars, bssn_d1,
                              rho0, h, W, vr, spacetime_mode, r):
        """
        Calcula fuentes físicas (no densitizadas) y luego multiplica por J_cell.
        Incluye:
          * término geométrico esférico en la ecuación de momento (+2p/r),
          * acoplos con gradientes de lapse/shift/φ (si spacetime es dinámico),
          * término de energía α K (p − T00) (ya con α dentro de S_phys).
        """
        N = len(r)
        r_safe = np.maximum(np.asarray(r, dtype=float), 1e-30)

        # Fuentes "físicas" (no densitizadas)
        src_Sr_phys  = np.zeros(N)
        src_tau_phys = np.zeros(N)

        # "Hoop stress" en 1D esférico (momento): +2p/r
        src_Sr_phys += 2.0 * pressure / r_safe

        if spacetime_mode != "fixed_minkowski":
            alpha = g['alpha']

            # T^{μν} (fluido perfecto)
            T00 = rho0 * h * W**2 - pressure
            T0r = rho0 * h * W**2 * vr
            Trr = rho0 * h * W**2 * vr**2 + pressure

            # Derivadas de campos BSSN
            dalpha_dr = bssn_d1.lapse[:, i_r]
            dbeta_dr  = bssn_d1.shift_U[:, i_r, i_r]
            dphi_dr   = bssn_d1.phi[:, i_r]

            # Momento: términos típicos (forma compacta 1D)
            src_Sr_phys += - T00 * dalpha_dr
            src_Sr_phys += + T0r * dbeta_dr

            # Aproximación 1D para ∂_r γ_rr (si necesitas exactitud total, cámbialo por tu cálculo completo)
            d_gamma_rr_dr = g['gamma_rr'] * (4.0 * dphi_dr)
            src_Sr_phys += 0.5 * Trr * d_gamma_rr_dr

            # Energía: término estándar (α K (p − T00))
            src_tau_phys += alpha * bssn_vars.K * (pressure - T00)

        # Densitización de las fuentes con J_cell  (NO multiplicar por α aquí)
        J_cell = g['J_cell']
        src_Sr  = J_cell * src_Sr_phys
        src_tau = J_cell * src_tau_phys
        return src_Sr, src_tau


    # ------------------------------------------------------------------
    # FUENTES (FULL):  J · S_phys
    # ------------------------------------------------------------------
    def _compute_sources_full(self, pressure, g, bssn_vars, bssn_d1,
                            rho0, h, W, vr, spacetime_mode, r):
        """
        Implementa ~s del paper (ec. (34) para el momento y (47) para energía)
        densitizadas con J (ver (53)). En Minkowski fijo sobre coordenadas
        esféricas, el único término no nulo es el “hoop stress” +2p/r en Sr.
        """
        N = len(r)
        J_cell = g['J_cell']

        # --- caso Minkowski fijo: solo hoop stress en la ecuación de momento
        if spacetime_mode == "fixed_minkowski":
            r_safe = np.maximum(np.abs(np.asarray(r, dtype=float)), 1e-30)
            src_Sr_phys  = 2.0 * pressure / r_safe   # Γ̂^θ_{rθ} = Γ̂^φ_{rφ} = 1/r  ⇒  +2p/r
            src_tau_phys = np.zeros(N)               # no hay fuentes geométricas en τ con Cowling
            return J_cell * src_Sr_phys, J_cell * src_tau_phys

        # --- caso métrica dinámica: términos completos de (34) y (47)
        alpha    = g['alpha']
        gamma_rr = g['gamma_rr']

        # T^{μν} para fluido perfecto
        T00 = rho0 * h * W**2 - pressure
        T0r = rho0 * h * W**2 * vr
        Trr = rho0 * h * W**2 * vr**2 + pressure

        # Derivadas BSSN (radial)
        dalpha_dr = bssn_d1.lapse[:, i_r]            # ∂r α
        dbeta_dr  = bssn_d1.shift_U[:, i_r, i_r]     # ∂r β^r   (en tu notación)
        dphi_dr   = bssn_d1.phi[:, i_r]              # ∂r φ
        # Aproximación 1D para ∂r γ_rr a partir de φ (γ_rr = e^{4φ} \barγ_rr, con \barγ_rr≈1)
        d_gamma_rr_dr = gamma_rr * (4.0 * dphi_dr)

        # (sS)^r (ec. (34)), escrito en términos de componentes útiles en 1D
        src_Sr_phys  = (- T00 * dalpha_dr) + (T0r * dbeta_dr) + 0.5 * Trr * d_gamma_rr_dr

        # s_τ (ec. (47)) – usa K y gradientes de α (no incluir términos duplicados)
        src_tau_phys = alpha * bssn_vars.K * (pressure - T00)  \
                    + (T00 * (- bssn_d1.lapse[:, i_r] * 0.0))  # placeholder si más términos de gauge

        return J_cell * src_Sr_phys, J_cell * src_tau_phys

    # ------------------------------------------------------------------
    # BOUNDARIES (con 3 ghost layers)
    # ------------------------------------------------------------------
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
            D[i]   =  D[mirror]     # par
            Sr[i]  = -Sr[mirror]    # impar
            tau[i] =  tau[mirror]   # par

        # frontera externa (lado derecho)
        last_interior = N - NUM_GHOSTS - 1
        for k in range(1, NUM_GHOSTS + 1):
            idx = last_interior + k
            D[idx]   = D[last_interior]
            Sr[idx]  = Sr[last_interior]
            tau[idx] = tau[last_interior]
        return D, Sr, tau

    def _apply_rhs_boundary_conditions(self, rhs_D, rhs_Sr, rhs_tau, r):
        """Las mismas paridades para el RHS."""
        N = len(r)

        # r=0
        for i in range(NUM_GHOSTS):
            mirror = 2 * NUM_GHOSTS - 1 - i
            rhs_D[i]   =  rhs_D[mirror]     # par
            rhs_Sr[i]  = -rhs_Sr[mirror]    # impar
            rhs_tau[i] =  rhs_tau[mirror]   # par

        # frontera externa
        last_interior = N - NUM_GHOSTS - 1
        for k in range(1, NUM_GHOSTS + 1):
            idx = last_interior + k
            rhs_D[idx]   = rhs_D[last_interior]
            rhs_Sr[idx]  = rhs_Sr[last_interior]
            rhs_tau[idx] = rhs_tau[last_interior]
        return rhs_D, rhs_Sr, rhs_tau
