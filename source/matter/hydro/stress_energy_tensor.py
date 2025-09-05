import numpy as np
from bssn.tensoralgebra import *
from bssn.tensoralgebra import SPACEDIM
from core.grid import i_x1, i_x2, i_x3
from bssn.bssnstatevariables import NUM_BSSN_VARS

def compute_stress_energy_tensor(rho0, vr, pressure, eps, W, h, r, bssn_vars, background, spacetime_mode):
    """
    Compute stress-energy tensor T^μν for relativistic perfect fluid.
    
    Perfect fluid stress-energy tensor:
    T^μν = ρ₀ h u^μ u^ν + p g^μν
    
    where:
    - ρ₀ = rest mass density
    - h = specific enthalpy = 1 + ε + p/ρ₀
    - u^μ = four-velocity
    - p = pressure
    - g^μν = inverse spacetime metric
    
    Returns EMTensor object with components needed by engrenage BSSN evolution:
    - rho: Energy density T^tt
    - Si: Momentum density T^ti
    - Sij: Stress tensor T^ij
    - S: Trace S = γ_ij T^ij
    """
    
    N = np.size(r)
    emtensor = EMTensor(N)
    
    # Extract or compute geometric quantities
    alpha, beta_r, gamma_LL, gamma_UU = _extract_geometry(r, bssn_vars, background, spacetime_mode)
    
    # Compute four-velocity components
    ut, ur, ut_cov, ur_cov = _compute_four_velocity(rho0, vr, W, alpha, beta_r, gamma_LL)
    
    # Compute stress-energy tensor components
    _compute_energy_density(emtensor, rho0, h, ut, ur, pressure, alpha, beta_r, gamma_UU)
    _compute_momentum_density(emtensor, rho0, h, ut, ur, pressure, alpha, beta_r, gamma_UU)
    _compute_stress_tensor(emtensor, rho0, h, ur, pressure, gamma_UU)
    _compute_stress_trace(emtensor, gamma_LL)
    
    # Validate physical consistency
    _validate_stress_energy_tensor(emtensor, rho0, pressure, W, r)
    
    return emtensor


def _extract_geometry(r, bssn_vars, background, spacetime_mode):
    """Extract geometric quantities from BSSN variables or use Minkowski."""
    
    N = len(r)
    
    if spacetime_mode == "fixed_minkowski":
        # Static Minkowski metric in spherical coordinates
        # ds² = -dt² + dr² + r²dθ² + r²sin²θdφ²
        alpha = np.ones(N)
        beta_r = np.zeros(N)
        
        # Spatial metric γ_ij = diag(1, r², r²sin²θ)
        gamma_LL = np.zeros([N, SPACEDIM, SPACEDIM])
        gamma_LL[:, 0, 0] = 1.0        # γ_rr = 1
        gamma_LL[:, 1, 1] = r**2       # γ_θθ = r²
        gamma_LL[:, 2, 2] = r**2       # γ_φφ = r² (sin²θ will be handled separately)
        
        # Inverse spatial metric γ^ij
        gamma_UU = np.zeros([N, SPACEDIM, SPACEDIM])
        gamma_UU[:, 0, 0] = 1.0        # γ^rr = 1
        gamma_UU[:, 1, 1] = 1.0/r**2   # γ^θθ = 1/r²
        gamma_UU[:, 2, 2] = 1.0/r**2   # γ^φφ = 1/r² (sin⁻²θ will be handled separately)
        
    else:
        # Extract from BSSN variables for dynamic spacetime
        alpha = bssn_vars.lapse
        beta_r = bssn_vars.shift_U[:, i_x1] if hasattr(bssn_vars, 'shift_U') else np.zeros(N)
        
        # Reconstruct spatial metric from conformal BSSN variables
        # γ_ij = e^{4φ} γ̄_ij
        e4phi = np.exp(4.0 * bssn_vars.phi)
        bar_gamma_LL = get_bar_gamma_LL(r, bssn_vars.h_LL, background)
        
        gamma_LL = np.zeros([N, SPACEDIM, SPACEDIM])
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                gamma_LL[:, i, j] = e4phi * bar_gamma_LL[:, i, j]
        
        # Inverse spatial metric γ^ij = e^{-4φ} γ̄^ij
        bar_gamma_UU = get_bar_gamma_UU(r, bssn_vars.h_LL, background)
        em4phi = np.exp(-4.0 * bssn_vars.phi)
        
        gamma_UU = np.zeros([N, SPACEDIM, SPACEDIM])
        for i in range(SPACEDIM):
            for j in range(SPACEDIM):
                gamma_UU[:, i, j] = em4phi * bar_gamma_UU[:, i, j]
    
    return alpha, beta_r, gamma_LL, gamma_UU


def _compute_four_velocity(rho0, vr, W, alpha, beta_r, gamma_LL):
    """
    Compute four-velocity components u^μ and u_μ.
    
    Four-velocity normalization: g_μν u^μ u^ν = -1
    In spherical symmetry with only radial motion:
    - u^t = W/α (timelike component)
    - u^r = W v^r (radial component)  
    - u^θ = u^φ = 0 (no angular motion)
    """
    
    # Contravariant four-velocity
    ut = W / alpha                    # u^t = W/α
    ur = W * vr                       # u^r = W v^r
    
    # Covariant four-velocity u_μ = g_μν u^ν
    # u_t = g_tt u^t + g_tr u^r = -α² u^t + γ_rr β^r u^r
    # u_r = g_rt u^t + g_rr u^r = γ_rr β^r u^t + γ_rr u^r
    gamma_rr = gamma_LL[:, 0, 0]
    
    ut_cov = -alpha**2 * ut + gamma_rr * beta_r * ur
    ur_cov = gamma_rr * beta_r * ut + gamma_rr * ur
    
    return ut, ur, ut_cov, ur_cov


def _compute_energy_density(emtensor, rho0, h, ut, ur, pressure, alpha, beta_r, gamma_UU):
    """
    Compute energy density ρ = n_μ n_ν T^μν where n^μ is normal to slice.
    
    For perfect fluid: ρ = T^μν n_μ n_ν = (ρ₀h W² - p)
    """
    
    # Method 1: Direct formula (preferred for stability)
    # W² = (u^t * α)²
    W2 = (ut * alpha)**2
    emtensor.rho = rho0 * h * W2 - pressure
    
    # Alternative Method 2: Component calculation (commented out as per recommendation)
    # This would compute T^tt and project with normal vectors
    # gamma_UU_rr = gamma_UU[:, 0, 0]
    # g_UU_tt = -1.0/alpha**2 + gamma_UU_rr * beta_r**2/alpha**2
    # g_UU_tr = gamma_UU_rr * beta_r/alpha**2
    # 
    # T_UU_tt = rho0 * h * ut**2 + pressure * g_UU_tt
    # T_UU_tr = rho0 * h * ut * ur + pressure * g_UU_tr
    # T_UU_rr = rho0 * h * ur**2 + pressure * gamma_UU_rr
    # 
    # emtensor.rho = alpha**2 * T_UU_tt - 2*alpha*beta_r*T_UU_tr + beta_r**2*gamma_UU_rr*T_UU_rr
    
    # Ensure positivity (should be automatic for physical states)
    emtensor.rho = np.maximum(emtensor.rho, 0.0)


def _compute_momentum_density(emtensor, rho0, h, ut, ur, pressure, alpha, beta_r, gamma_UU):
    """
    Compute momentum density S_i = T^t_i.
    
    In spherical symmetry:
    S_r = T^t_r = ρ₀ h u^t u^r + p g^tr
    S_θ = S_φ = 0 (no angular momentum)
    """
    
    N = len(rho0)
    emtensor.Si = np.zeros([N, SPACEDIM])
    
    # Radial momentum density
    # g^tr = γ^rr β^r/α² for our coordinate system
    gamma_UU_rr = gamma_UU[:, 0, 0]
    g_UU_tr = gamma_UU_rr * beta_r / alpha**2
    
    Sr_density = rho0 * h * ut * ur + pressure * g_UU_tr
    emtensor.Si[:, i_x1] = Sr_density
    
    # Angular components are zero in spherical symmetry
    emtensor.Si[:, i_x2] = 0.0  # S_θ = 0
    emtensor.Si[:, i_x3] = 0.0  # S_φ = 0


def _compute_stress_tensor(emtensor, rho0, h, ur, pressure, gamma_UU):
    """
    Compute stress tensor S^ij = T^ij.
    
    T^ij = ρ₀ h u^i u^j + p γ^ij
    
    In spherical symmetry with only radial motion:
    - T^rr = ρ₀ h (u^r)² + p γ^rr
    - T^θθ = p γ^θθ (no u^θ component)
    - T^φφ = p γ^φφ (no u^φ component)
    - Off-diagonal terms are zero
    """
    
    N = len(rho0)
    emtensor.Sij = np.zeros([N, SPACEDIM, SPACEDIM])
    
    # Diagonal components
    emtensor.Sij[:, i_x1, i_x1] = rho0 * h * ur**2 + pressure * gamma_UU[:, 0, 0]  # T^rr
    emtensor.Sij[:, i_x2, i_x2] = pressure * gamma_UU[:, 1, 1]                     # T^θθ
    emtensor.Sij[:, i_x3, i_x3] = pressure * gamma_UU[:, 2, 2]                     # T^φφ
    
    # Off-diagonal terms are zero in spherical symmetry with radial flow


def _compute_stress_trace(emtensor, gamma_LL):
    """
    Compute trace S = γ_ij T^ij.
    
    This is the trace of the spatial stress tensor.
    """
    
    # Contract spatial metric with stress tensor
    # S = γ_rr T^rr + γ_θθ T^θθ + γ_φφ T^φφ
    emtensor.S = (gamma_LL[:, 0, 0] * emtensor.Sij[:, 0, 0] +
                  gamma_LL[:, 1, 1] * emtensor.Sij[:, 1, 1] +
                  gamma_LL[:, 2, 2] * emtensor.Sij[:, 2, 2])


def _validate_stress_energy_tensor(emtensor, rho0, pressure, W, r):
    """
    Validate stress-energy tensor for physical consistency.
    
    Checks:
    - Energy density is positive
    - Dominant energy condition
    - Trace relationships
    - NaN/infinity checks
    """
    
    # Check for NaN or infinity
    if np.any(np.isnan(emtensor.rho)) or np.any(np.isinf(emtensor.rho)):
        print("Warning: NaN or infinity in energy density")
    
    if np.any(np.isnan(emtensor.S)) or np.any(np.isinf(emtensor.S)):
        print("Warning: NaN or infinity in stress trace")
    
    # Check energy density positivity
    negative_energy = emtensor.rho < 0
    if np.any(negative_energy):
        n_neg = np.sum(negative_energy)
        min_energy = np.min(emtensor.rho[negative_energy])
        print(f"Warning: {n_neg} points with negative energy density (min: {min_energy})")
    
    # Check dominant energy condition: |T^0i| ≤ T^00
    # For our case: |S_r| ≤ ρ
    violates_dec = np.abs(emtensor.Si[:, 0]) > emtensor.rho
    if np.any(violates_dec):
        n_violations = np.sum(violates_dec)
        print(f"Warning: {n_violations} points violate dominant energy condition")
    
    # Check for extremely large values (may indicate numerical issues)
    large_energy = emtensor.rho > 1e10 * np.max(rho0)
    if np.any(large_energy):
        n_large = np.sum(large_energy)
        max_ratio = np.max(emtensor.rho / rho0)
        print(f"Warning: {n_large} points with very large energy density (max ratio: {max_ratio})")
    
    # Trace consistency check for ideal gas
    # For ideal gas: T^μ_μ = ρ₀(1+ε) - 3p = ρ₀ + p/(γ-1) - 3p (for γ-law EOS)
    # This should be Lorentz invariant
    
    # Basic sanity check: stress trace should be reasonable
    expected_trace_magnitude = 3.0 * pressure  # Order of magnitude estimate
    large_trace = np.abs(emtensor.S) > 100.0 * expected_trace_magnitude
    if np.any(large_trace):
        n_large_trace = np.sum(large_trace)
        print(f"Warning: {n_large_trace} points with unexpectedly large stress trace")


def compute_stress_energy_divergence(emtensor, rho0, vr, pressure, W, h, 
                                   r, bssn_vars, bssn_d1, background, spacetime_mode):
    """
    Compute divergence ∇_μ T^μν for conservation law verification.
    
    Should be zero for a perfect fluid: ∇_μ T^μν = 0
    
    This function is useful for code validation and debugging.
    """
    
    N = len(r)
    
    # Need derivatives of primitive variables
    # This would require storing them in the fluid class
    # For now, approximate using finite differences
    
    dr = r[1] - r[0] if N > 1 else 1e-10
    
    # Radial derivatives (simple centered differences)
    d_rho0_dr = np.gradient(rho0, dr)
    d_pressure_dr = np.gradient(pressure, dr)
    d_vr_dr = np.gradient(vr, dr)
    
    # Get geometry
    alpha = bssn_vars.lapse
    d_alpha_dr = bssn_d1.lapse[:, 0] if hasattr(bssn_d1, 'lapse') else np.zeros(N)
    
    # Energy equation: ∇_μ T^μ0 = 0
    # Simplified for spherical symmetry, neglecting time derivatives
    div_energy = (
        -emtensor.rho * bssn_vars.K * alpha  # Expansion term
        + d_pressure_dr * vr * W**2          # Pressure gradient
        + 2.0 * pressure * vr * W**2 / r     # Geometric source
    )
    
    # Momentum equation: ∇_μ T^μr = 0  
    div_momentum_r = (
        d_pressure_dr                         # Pressure gradient
        + rho0 * h * W**2 * d_alpha_dr / alpha  # Gravitational acceleration
        + 2.0 * pressure / r                   # Geometric source (hoop stress)
    )
    
    return div_energy, div_momentum_r


def stress_energy_eigenvalues(emtensor, gamma_LL):
    """
    Compute eigenvalues of stress tensor T^ij for stability analysis.
    
    The eigenvalues give information about principal stresses
    and can be used to check for superluminal signal propagation.
    """
    
    N = len(emtensor.rho)
    eigenvals = np.zeros([N, SPACEDIM])
    
    for i in range(N):
        # Extract 3x3 stress tensor at point i
        Tij = emtensor.Sij[i, :, :]
        gamma_inv = np.linalg.inv(gamma_LL[i, :, :])
        
        # Compute T^ij = γ^ik γ^jl T_kl (raise indices)
        Tij_up = np.dot(gamma_inv, np.dot(Tij, gamma_inv))
        
        try:
            # Eigenvalues of T^ij
            vals = np.linalg.eigvals(Tij_up)
            eigenvals[i, :] = np.real(vals)  # Should be real for physical tensors
        except np.linalg.LinAlgError:
            # Handle singular matrices
            eigenvals[i, :] = 0.0
    
    return eigenvals


def compute_fluid_rest_frame_quantities(rho0, vr, pressure, eps, W, h):
    """
    Compute stress-energy tensor components in fluid rest frame.
    
    In the fluid rest frame:
    - T^tt = ρ₀(1 + ε) = ρ₀ + ρ₀ε = energy density
    - T^ij = p δ^ij = isotropic pressure
    - T^ti = 0 = no momentum density
    
    Useful for validation and physical interpretation.
    """
    
    # Energy density in rest frame
    energy_density_rest = rho0 * (1.0 + eps)
    
    # Pressure (isotropic)
    pressure_rest = pressure
    
    # Total energy density (including rest mass)
    total_energy_rest = rho0 + rho0 * eps
    
    return {
        'energy_density': energy_density_rest,
        'pressure': pressure_rest,
        'total_energy': total_energy_rest,
        'rest_mass_density': rho0,
        'internal_energy_density': rho0 * eps
    }