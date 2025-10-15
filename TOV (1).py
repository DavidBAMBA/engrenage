import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

##############################################
# Global (physical) parameters
##############################################
fluid_gamma = 2.0       # Adiabatic index
fluid_kappa = 1e2       # Polytropic constant
TOV_rho0    = 0.00128   # Central density for the TOV star
fluid_atmos = 1e-8      # Atmosphere density cutoff
smallpi = np.arccos(-1.0)

r_start = 1e-6          # Start radius (avoid r=0 singularity)
r_end   = 50.0          # End radius

##############################################
# Define the right-hand side of the ODE system
##############################################
def tov_system(r, y):
    """
    Computes the derivatives [dA/dr, d(rho0)/dr] for the TOV equations.
    When rho0 falls below the atmosphere threshold, its derivative is frozen.
    """
    A, rho0 = y
    eps = 1e-6
    r_safe = r if r > eps else eps

    # Specific internal energy: e = kappa/(gamma-1)*rho0^(gamma-1)
    e = fluid_kappa / (fluid_gamma - 1.0) * rho0**(fluid_gamma - 1.0)
    # Total energy density: rho_total = rho0*(1+e)
    rho_total = rho0 * (1.0 + e)
    # ODE for A(r): dA/dr = A * [ (1-A)/r + 8*pi*r*A*rho_total ]
    dA_dr = A * ((1.0 - A)/r_safe + 8.0 * np.pi * r_safe * A * rho_total)

    # Mass function: m(r) = (r/2) * (1 - 1/A)
    m = (r_safe / 2.0) * (1.0 - 1.0/A)
    # Term for drho0/dr
    term = (rho0**(1.0 - fluid_gamma) / fluid_kappa + fluid_gamma / (fluid_gamma - 1.0))
    if rho0 < fluid_atmos:
        drho0_dr = 0.0
    else:
        drho0_dr = -rho0 * term * (m / r_safe**2 + 4.0*np.pi*r_safe*fluid_kappa*rho0**fluid_gamma) \
                   / (fluid_gamma * (1.0 - 2.0*m / r_safe))
    return [dA_dr, drho0_dr]

##############################################
# Reference solution (very strict tolerances)
##############################################
sol_ref = solve_ivp(tov_system, [r_start, r_end], [1.0, TOV_rho0],
                    method='Radau', dense_output=True,
                    rtol=1e-12, atol=1e-14)
r_ref = np.linspace(r_start, r_end, 5000)
y_ref = sol_ref.sol(r_ref)
A_ref = y_ref[0]
rho0_ref = y_ref[1]

##############################################
# 1. Manual RK4 integration (self-contained)
##############################################
def manual_RK4(r_start, r_end, dr):
    """
    Integrates the TOV system using a fixed-step fourth-order Runge-Kutta method.
    Returns arrays for r, A(r), and rho0(r).
    """
    N = int((r_end - r_start)/dr) + 1
    r_arr = np.linspace(r_start, r_end, N)
    y = np.zeros((N, 2))
    y[0, :] = [1.0, TOV_rho0]  # initial conditions: A(0)=1, rho0(0)=TOV_rho0
    for i in range(N - 1):
        r_i = r_arr[i]
        y_i = y[i, :]
        # For the first step, use a half-step to alleviate the singularity at r=0.
        h = dr if i > 0 else dr/2.0
        k1 = np.array(tov_system(r_i, y_i))
        k2 = np.array(tov_system(r_i + h/2.0, y_i + (h/2.0)*k1))
        k3 = np.array(tov_system(r_i + h/2.0, y_i + (h/2.0)*k2))
        k4 = np.array(tov_system(r_i + h, y_i + h*k3))
        y[i+1, :] = y_i + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return r_arr, y[:,0], y[:,1]

##############################################
# 2. Integration using SciPy's solve_ivp (Radau)
##############################################
def ivp_integration(r_start, r_end, rtol, atol):
    """
    Solves the TOV system using solve_ivp with given tolerances.
    Returns arrays for r, A(r), and rho0(r) evaluated on a dense grid.
    """
    sol = solve_ivp(tov_system, [r_start, r_end], [1.0, TOV_rho0],
                    method='Radau', dense_output=True, rtol=rtol, atol=atol)
    r_dense = np.linspace(r_start, r_end, 5000)
    y_dense = sol.sol(r_dense)
    return r_dense, y_dense[0], y_dense[1]

##############################################
# Run the integrations at three resolutions
##############################################
# For manual RK4, we use three step sizes:
manual_steps = [0.005, 0.0025, 0.00125]
print("Manual RK4 convergence:")
for dr in manual_steps:
    r_manual, A_manual, rho0_manual = manual_RK4(r_start, r_end, dr)
    # Interpolate onto the reference grid.
    interp_A = interp1d(r_manual, A_manual, kind='linear', fill_value="extrapolate")
    interp_rho0 = interp1d(r_manual, rho0_manual, kind='linear', fill_value="extrapolate")
    A_manual_interp = interp_A(r_ref)
    rho0_manual_interp = interp_rho0(r_ref)
    error_A = np.linalg.norm(A_manual_interp - A_ref) / np.linalg.norm(A_ref)
    error_rho0 = np.linalg.norm(rho0_manual_interp - rho0_ref) / np.linalg.norm(rho0_ref)
    print(f"  dr = {dr:.5f} --> Relative L2 error in A(r): {error_A:.3e}, rho0(r): {error_rho0:.3e}")

# For solve_ivp, we vary the tolerances:
ivp_tol = [(1e-8,1e-10), (1e-10,1e-12), (1e-12,1e-14)]
print("\nsolve_ivp convergence:")
for rtol, atol in ivp_tol:
    r_ivp, A_ivp, rho0_ivp = ivp_integration(r_start, r_end, rtol, atol)
    error_A = np.linalg.norm(A_ivp - A_ref) / np.linalg.norm(A_ref)
    error_rho0 = np.linalg.norm(rho0_ivp - rho0_ref) / np.linalg.norm(rho0_ref)
    print(f"  rtol={rtol:.1e}, atol={atol:.1e} --> Relative L2 error in A(r): {error_A:.3e}, rho0(r): {error_rho0:.3e}")

##############################################
# Plotting: Compare best manual and ivp solutions vs. reference
##############################################
# Use the finest manual resolution and the strictest ivp tolerances.
r_manual, A_manual, rho0_manual = manual_RK4(r_start, r_end, manual_steps[-1])
r_ivp, A_ivp, rho0_ivp = ivp_integration(r_start, r_end, ivp_tol[-1][0], ivp_tol[-1][1])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(r_ref, A_ref, 'k-', label='Reference A(r)')
plt.plot(r_manual, A_manual, 'b--', label=f'Manual RK4 (dr={manual_steps[-1]})')
plt.plot(r_ivp, A_ivp, 'r-.', label='solve_ivp (strict tol)')
plt.xlabel('r')
plt.ylabel('A(r)')
plt.title('Comparison of A(r)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(r_ref, rho0_ref, 'k-', label='Reference ρ₀(r)')
plt.plot(r_manual, rho0_manual, 'b--', label=f'Manual RK4 (dr={manual_steps[-1]})')
plt.plot(r_ivp, rho0_ivp, 'r-.', label='solve_ivp (strict tol)')
plt.xlabel('r')
plt.ylabel('ρ₀(r)')
plt.title('Comparison of ρ₀(r)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

