import numpy as np
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import i_r, i_t, i_p

class ValenciaReferenceMetric:
    """
    Valencia formulation with reference metric for spherical coordinates.
    Following Montero et al. (2013) - arXiv:1309.7808
    
    This class implements the core physics of the Valencia formulation
    in the reference-metric approach for curvilinear coordinates.
    """
    
    def __init__(self):
        self.name = "valencia_reference_metric"
        self.version = "montero_2013"
        
    def compute_rhs(self, D, Sr, tau, rho0, vr, pressure, W, h,
                   d1_D, d1_Sr, d1_tau, r, bssn_vars, bssn_d1, background,
                   spacetime_mode, eos, reconstructor=None, riemann_solver=None):
        """
        Compute RHS for the Valencia evolution equations with reference metric.
        
        Evolution equations (Montero et al. 2013):
        ∂_t q + D̂_j f^j = s
        where q = e^(6φ)√(γ̄/γ̂) (D, S_i, τ)
        
        Args:
            D, Sr, tau: Conservative variables
            rho0, vr, pressure, W, h: Primitive variables
            d1_D, d1_Sr, d1_tau: First derivatives of conservative vars
            r: Radial coordinate array
            bssn_vars: BSSN variables object
            bssn_d1: BSSN first derivatives
            background: Reference metric background
            spacetime_mode: "fixed_minkowski" or "dynamic"
            eos: Equation of state object
            reconstructor: Reconstruction method (optional)
            riemann_solver: Riemann solver (optional)
            
        Returns:
            dDdt, dSrdt, dtaudt: Time derivatives
        """
        
        N = len(r)
        
        # Extract geometry
        geometry = self._extract_geometry(r, bssn_vars, spacetime_mode, background)
        
        # Compute fluxes
        if riemann_solver is not None and reconstructor is not None:
            fluxes = self._compute_interface_fluxes(
                D, Sr, tau, rho0, vr, pressure, W, h,
                geometry, r, eos, reconstructor, riemann_solver
            )
            
            # Compute covariant divergence
            dDdt = self._covariant_divergence_D(fluxes['D'], geometry, background)
            dSrdt_flux = self._covariant_divergence_Sr(fluxes['Sr'], geometry, background)
            dtaudt = self._covariant_divergence_tau(fluxes['tau'], geometry, background)
        else:
            # Cell-centered fluxes
            fluxes = self._compute_cell_fluxes(
                D, Sr, tau, rho0, vr, pressure, W, h, geometry
            )
            
            dDdt = self._continuity_rhs(fluxes['D'], geometry, background)
            dSrdt_flux = self._momentum_flux_rhs(fluxes['Sr'], geometry, background)
            dtaudt = self._energy_rhs(fluxes['tau'], geometry, background)
        
        # Compute source terms
        sources = self._compute_sources(
            D, Sr, tau, rho0, vr, pressure, W, h,
            geometry, bssn_vars, bssn_d1, spacetime_mode, background
        )
        
        # Combine flux divergence and sources
        dDdt = dDdt
        dSrdt = dSrdt_flux + sources['Sr']
        dtaudt = dtaudt + sources['tau']
        
        # Apply boundary conditions
        dDdt, dSrdt, dtaudt = self._apply_boundary_conditions(
            dDdt, dSrdt, dtaudt, r
        )
        
        return dDdt, dSrdt, dtaudt
    
    def _extract_geometry(self, r, bssn_vars, spacetime_mode, background):
        """
        Extract all geometric quantities needed for Valencia evolution.
        
        Returns dictionary with:
            - Metric components (alpha, beta, gamma_ij)
            - Conformal factors (e4phi, e6phi)
            - Determinants (sqrt_gamma_bar, sqrt_gamma_hat, sqrt_gamma_ratio)
        """
        
        N = len(r)
        geometry = {}
        
        if spacetime_mode == "fixed_minkowski":
            geometry['alpha'] = np.ones(N)
            geometry['beta_r'] = np.zeros(N)
            geometry['e4phi'] = np.ones(N)
            geometry['e6phi'] = np.ones(N)
            geometry['gamma_rr'] = np.ones(N)
            geometry['gamma_tt'] = r**2
            geometry['gamma_pp'] = r**2
            
            # Bar metric = physical metric in Minkowski
            geometry['bar_gamma_LL'] = np.zeros([N, SPACEDIM, SPACEDIM])
            geometry['bar_gamma_LL'][:, 0, 0] = 1.0
            geometry['bar_gamma_LL'][:, 1, 1] = r**2
            geometry['bar_gamma_LL'][:, 2, 2] = r**2
            
            geometry['bar_gamma_UU'] = np.zeros([N, SPACEDIM, SPACEDIM])
            geometry['bar_gamma_UU'][:, 0, 0] = 1.0
            geometry['bar_gamma_UU'][:, 1, 1] = 1.0/r**2
            geometry['bar_gamma_UU'][:, 2, 2] = 1.0/r**2
            
        else:
            geometry['alpha'] = bssn_vars.lapse
            geometry['beta_r'] = bssn_vars.shift_U[:, i_r] if hasattr(bssn_vars, 'shift_U') else np.zeros(N)
            geometry['e4phi'] = np.exp(4.0 * bssn_vars.phi)
            geometry['e6phi'] = np.exp(6.0 * bssn_vars.phi)
            
            # Get conformal metric
            geometry['bar_gamma_LL'] = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
            geometry['bar_gamma_UU'] = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
            
            # Physical metric components
            geometry['gamma_rr'] = geometry['e4phi'] * geometry['bar_gamma_LL'][:, i_r, i_r]
            geometry['gamma_tt'] = geometry['e4phi'] * geometry['bar_gamma_LL'][:, i_t, i_t]
            geometry['gamma_pp'] = geometry['e4phi'] * geometry['bar_gamma_LL'][:, i_p, i_p]
        
        # Reference metric determinant (spherical coordinates)
        geometry['sqrt_gamma_hat'] = r**2  # For spherical coords: √γ̂ = r² (assuming sin θ = 1 at equator)
        geometry['sqrt_gamma_bar'] = r**2  # For spherical symmetry
        
        # Critical reference-metric factor
        geometry['sqrt_gamma_ratio'] = geometry['e6phi'] * (
                geometry['sqrt_gamma_bar'] / np.maximum(geometry['sqrt_gamma_hat'], 1e-20))
        

        
        # Store radial coordinate for convenience
        geometry['r'] = r
        
        return geometry
    
    def _compute_cell_fluxes(self, D, Sr, tau, rho0, vr, pressure, W, h, geometry):
        """
        Compute fluxes at cell centers (Eqs. 18, 27, 42 of Montero et al.)
        
        Returns dictionary with flux components.
        """
        
        alpha = geometry['alpha']
        beta_r = geometry['beta_r']
        sqrt_gamma_ratio = geometry['sqrt_gamma_ratio']
        
        # Transport velocity
        vr_transport = vr - beta_r/alpha
        
        fluxes = {}
        
        # Continuity flux (Eq. 18)
        fluxes['D'] = alpha * sqrt_gamma_ratio * D * vr_transport
        
        # Momentum flux (Eq. 27)
        T_rr = W**2 * rho0 * h * vr**2 + pressure
        fluxes['Sr'] = alpha * sqrt_gamma_ratio * T_rr
        
        # Energy flux (Eq. 42)
        fluxes['tau'] = alpha * sqrt_gamma_ratio * (tau + pressure) * vr_transport
        
        return fluxes
    

    def _continuity_rhs(self, flux_D, geometry, background):
        r = geometry["r"]
        N = len(r)
        
        # Usar el sistema de derivadas de Grid de Engrenage
        if hasattr(self, 'grid'):
            # Crear estado temporal para usar el sistema de derivadas de Grid
            flux_state = np.zeros((1, N))
            flux_state[0] = flux_D
            d_flux_dr = self.grid.get_first_derivative(flux_state, [0])[0]
            dDdt = -d_flux_dr
        else:
            # Fallback si no hay acceso a Grid
            dr = r[1:] - r[:-1]
            dDdt = np.zeros(N)
            dDdt[1:-1] = -(flux_D[2:] - flux_D[:-2]) / (r[2:] - r[:-2])
            dDdt[0] = -(flux_D[1] - flux_D[0]) / (r[1] - r[0])
            dDdt[-1] = -(flux_D[-1] - flux_D[-2]) / (r[-1] - r[-2])
        
        # Término geométrico
        r_safe = np.maximum(r, 1e-10)
        dDdt -= 2.0 * flux_D / r_safe
        
        # Regularización en el origen
        if r[0] < 1e-10:
            dDdt[0] = -3.0 * (flux_D[1] - flux_D[0]) / (r[1] - r[0])
        
        return dDdt


    def _momentum_flux_rhs(self, flux_Sr, geometry, background):
        """
        ∂_t(e^{6φ} √(γ̄/γ̂) S_r) + ∂_r(f_S)_r^r
        = (f_S)_k^r Γ̂^k_{r r} - (f_S)_r^k Γ̂^j_{k j}.
        In spherical symmetry with only radial flux: Γ̂^k_{r r}=0 and Γ̂^j_{k j}=2/r,
        so the geometric source reduces to  -(f_S)_r^r * (2/r).
        """
        r = geometry["r"]
        N = len(r)
        if N < 2:
            return np.zeros(N)

        # r-aware divergence of flux
        dSrdt = -np.gradient(flux_Sr, r, edge_order=2)

        # Reference-metric source: -(f_S)_r^r * (2/r)
        r_safe = np.maximum(r, 1e-10)
        dSrdt -= 2.0 * flux_Sr / r_safe

        # Symmetry at the origin (Sr should vanish at r=0 in spherical symmetry)
        if r[0] < 1e-10:
            dSrdt[0] = 0.0

        return dSrdt


    def _energy_rhs(self, flux_tau, geometry, background):
        """
        ∂_t(e^{6φ} √(γ̄/γ̂) τ) + ∂_r(f_τ)^r = -(f_τ)^r Γ̂^k_{rk},   with Γ̂^k_{rk} = 2/r.
        """
        r = geometry["r"]
        N = len(r)
        if N < 2:
            return np.zeros(N)

        # r-aware divergence of flux
        dtaudt = -np.gradient(flux_tau, r, edge_order=2)

        # Reference-metric source: -(f_τ)^r * (2/r)
        r_safe = np.maximum(r, 1e-10)
        dtaudt -= 2.0 * flux_tau / r_safe

        # Regularize the origin (L'Hôpital/Taylor)
        if r[0] < 1e-10:
            dtaudt[0] = -3.0 * (flux_tau[1] - flux_tau[0]) / (r[1] - r[0])

        return dtaudt


    def _covariant_divergence_D(self, flux_D, geometry, background):
        """
        Compute covariant divergence D̂_j f^j for density flux.
        This is a wrapper that calls the appropriate RHS computation.
        """
        return self._continuity_rhs(flux_D, geometry, background)
    
    def _covariant_divergence_Sr(self, flux_Sr, geometry, background):
        """
        Compute covariant divergence D̂_j f^j for momentum flux.
        This is a wrapper that calls the appropriate RHS computation.
        """
        return self._momentum_flux_rhs(flux_Sr, geometry, background)
    
    def _covariant_divergence_tau(self, flux_tau, geometry, background):
        """
        Compute covariant divergence D̂_j f^j for energy flux.
        This is a wrapper that calls the appropriate RHS computation.
        """
        return self._energy_rhs(flux_tau, geometry, background)
    
    def _compute_interface_fluxes(self, D, Sr, tau, rho0, vr, pressure, W, h,
                                geometry, r, eos, reconstructor, riemann_solver):
        N = len(r)
        alpha = geometry['alpha']
        beta_r = geometry['beta_r']
        gamma_rr = geometry['gamma_rr']
        sqrt_gamma_ratio = geometry['sqrt_gamma_ratio']

        # 1) Reconstruct primitives to interfaces
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.reconstruct_primitive_variables(
            rho0, vr, pressure, x=r, boundary_type="outflow"
        )
        (rhoL, vL, pL), (rhoR, vR, pR) = reconstructor.apply_physical_limiters(
            (rhoL, vL, pL), (rhoR, vR, pR)
        )

        # 2) Metric at interfaces (simple average)
        gamma_rr_face = 0.5 * (gamma_rr[:-1] + gamma_rr[1:])
        alpha_face    = 0.5 * (alpha[:-1]    + alpha[1:])
        beta_r_face   = 0.5 * (beta_r[:-1]   + beta_r[1:])
        sg_ratio_face = 0.5 * (sqrt_gamma_ratio[:-1] + sqrt_gamma_ratio[1:])

        # 3) Build conservative states from primitives (non-densitized)
        def prim2cons(rho, v, p, g_rr):
            W = 1.0 / np.sqrt(1.0 - g_rr * v*v)
            eps = eos.eps_from_rho_p(rho, p)
            h   = 1.0 + eps + p / rho
            D   = rho * W
            Sr  = rho * h * W*W * (g_rr * v)    # v_r = γ_rr v^r
            tau = rho * h * W*W - p - D
            return D, Sr, tau

        DL, SrL, tauL = prim2cons(rhoL[:-1], vL[:-1], pL[:-1], gamma_rr_face)
        DR, SrR, tauR = prim2cons(rhoR[1:],  vR[1:],  pR[1:],  gamma_rr_face)

        # 4) HLLE per interface (non-densitized flux)
        F_D   = np.empty(N-1); F_Sr = np.empty(N-1); F_tau = np.empty(N-1)
        for i in range(N-1):
            UL = np.array([DL[i], SrL[i], tauL[i]])
            UR = np.array([DR[i], SrR[i], tauR[i]])
            primL = (rhoL[i], vL[i], pL[i])
            primR = (rhoR[i+1], vR[i+1], pR[i+1])
            F = riemann_solver.solve(UL, UR, primL, primR,
                                    gamma_rr_face[i], alpha_face[i], beta_r_face[i], eos)
            F_D[i], F_Sr[i], F_tau[i] = F

        # 5) Densitize for divergence
        F_D   *= alpha_face * sg_ratio_face
        F_Sr  *= alpha_face * sg_ratio_face
        F_tau *= alpha_face * sg_ratio_face

        # Return cell-centered flux arrays via simple face-to-cell mapping (e.g. average)
        fluxes = {
            'D':   np.concatenate([[F_D[0]],   0.5*(F_D[:-1]+F_D[1:]),   [F_D[-1]]]),
            'Sr':  np.concatenate([[F_Sr[0]],  0.5*(F_Sr[:-1]+F_Sr[1:]), [F_Sr[-1]]]),
            'tau': np.concatenate([[F_tau[0]], 0.5*(F_tau[:-1]+F_tau[1:]), [F_tau[-1]]])
        }
        return fluxes

    def _compute_sources(self, D, Sr, tau, rho0, vr, pressure, W, h,
                        geometry, bssn_vars, bssn_d1, spacetime_mode, background):
        """
        Compute source terms (Eqs. 34, 47 of Montero et al.)
        
        Returns dictionary with source components.
        """
        
        N = len(geometry['r'])
        sources = {'D': np.zeros(N), 'Sr': np.zeros(N), 'tau': np.zeros(N)}
        
        # Momentum source (Eq. 34)
        sources['Sr'] = self._momentum_source(
            pressure, geometry, bssn_vars, bssn_d1,
            rho0, h, W, vr, spacetime_mode, background
        )
        
        # Energy source (Eq. 47)
        if spacetime_mode != "fixed_minkowski" and hasattr(bssn_vars, 'K'):
            alpha = geometry['alpha']
            sqrt_gamma_ratio = geometry['sqrt_gamma_ratio']
            
            # Source term: s_τ = αe^6φ √(γ̄/γ̂) T^ab ∇_b n_a
            # From Eq. (47): includes K_ij terms
            sources['tau'] = alpha * sqrt_gamma_ratio * (
                -bssn_vars.K * (rho0 * h * W**2 - pressure)  # T^00 K term
            )
        
        return sources
    
    def _momentum_source(self, pressure, geometry, bssn_vars, bssn_d1,
                        rho0, h, W, vr, spacetime_mode, background):
        """
        Compute momentum source term following Eq. 34 of Montero et al.
        
        (s_S)_i = αe^6φ √(γ̄/γ̂) [T^ab Γ^a_bi - T^k_j Γ̂^k_ji]
        
        For spherical coordinates, the dominant term is the geometric source.
        """
        
        N = len(geometry['r'])
        r = geometry['r']
        alpha = geometry['alpha']
        sqrt_gamma_ratio = geometry['sqrt_gamma_ratio']
        
        # Initialize source
        source = np.zeros(N)
        
        # Geometric source from spherical coordinates
        # Main contribution: T^θθ Γ^r_θθ + T^φφ Γ^r_φφ = -r(T^θθ + T^φφ) = -2rp
        # This gives the 2p/r term after dividing by r
        r_safe = np.maximum(r, 1e-10)
        source = 2.0 * pressure / r_safe
        
        # Regularize at origin using L'Hôpital's rule
        if r[0] < 1e-10 and N > 1:
            dr = r[1] - r[0]
            dp_dr = (pressure[1] - pressure[0]) / dr
            source[0] = 2.0 * dp_dr
        
        # Add spacetime sources for dynamic case
        if spacetime_mode != "fixed_minkowski" and hasattr(bssn_d1, 'lapse'):
            # Stress-energy components
            T00 = rho0 * h * W**2 - pressure
            
            # Lapse gradient contribution: -T^00 α ∂_i α
            d_alpha_dr = bssn_d1.lapse[:, i_r]
            source += -T00 * d_alpha_dr
            
            # Additional metric derivative terms would go here
            # Following Eq. (34) of the paper
        
        # Scale by reference metric factor
        return alpha * sqrt_gamma_ratio * source
    
    def _apply_boundary_conditions(self, dDdt, dSrdt, dtaudt, r):
        """
        Apply spherical symmetry boundary conditions.
        
        At r=0: Use parity conditions
        At r_max: Use outflow conditions
        """
        
        # Inner boundary (r=0): parity conditions
        if r[0] < 1e-10:
            # Enforce only odd parity for Sr; keep D and tau from the regularized divergence
            dSrdt[0] = 0
        
        # Outer boundary: simple outflow (copy from interior)
        # This maintains the existing gradients
        if len(r) > 1:
            dDdt[-1] = dDdt[-2]
            dSrdt[-1] = dSrdt[-2]
            dtaudt[-1] = dtaudt[-2]
        
        return dDdt, dSrdt, dtaudt