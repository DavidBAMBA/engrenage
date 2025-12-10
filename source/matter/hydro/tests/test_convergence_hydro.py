#!/usr/bin/env python3
"""
Convergence test for relativistic hydrodynamics in Engrenage.
Tests advection of a Gaussian pulse with multiple grid resolutions and reconstructors.
Uses valencia.compute_rhs directly with fixed_minkowski spacetime.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import multiprocessing as mp
import pandas as pd

# Add source path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.core.spacing import NUM_GHOSTS
from source.matter.hydro.eos import IdealGasEOS
from source.matter.hydro.reconstruction import Reconstruction, create_reconstruction
from source.matter.hydro.riemann import HLLRiemannSolver
from source.matter.hydro.cons2prim import cons_to_prim
from source.matter.hydro.valencia_reference_metric import ValenciaReferenceMetric

# ── Global parameters ───────────────────────────────────────────────────
# Grid resolutions for convergence test
Ns = [50, 100, 200, 400, 800]

# Reconstructors to test
reconstructors = ["minmod", "mp5", "weno5", "wenoz"]

# Test parameters
velocity = 0.3       # Gaussian advection velocity (v_r < 1)
t_final = 0.3        # Final evolution time
gamma_gas = 1.4      # Ideal gas EOS gamma
cfl = 0.4            # CFL number

# Grid parameters (similar to test.py)
n_interior = 400     # Will vary for convergence test
r_min = 1e-3
r_max = 1.0

# Gaussian parameters
gauss_center = 0.5
gauss_width = 0.05
gauss_amplitude = 0.1
background_rho = 1e-3

# Setup data folders
os.makedirs("convergence_data", exist_ok=True)
os.makedirs("convergence_errors", exist_ok=True)

# ── Helper classes (similar to test.py) ────────────────────────────────────
class Grid:
    def __init__(self, dx):
        self.dr = float(dx)

class _DummyBSSNVars:
    """Placeholders no usados en Minkowski fijo."""
    def __init__(self, N):
        self.lapse   = np.ones(N)
        self.shift_U = np.zeros((N,3,3))
        self.phi     = np.zeros(N)
        self.K       = np.zeros(N)

class _DummyBSSND1:
    def __init__(self, N):
        self.lapse   = np.zeros((N,3))
        self.shift_U = np.zeros((N,3,3))
        self.phi     = np.zeros((N,3))

# ── Grid and primitive functions (from test.py) ─────────────────────────
def build_grid(n_interior=256, r_min=1.0e-3, r_max=1.0, ng=NUM_GHOSTS):
    """Centros de celda uniformes + ghosts extrapolados linealmente."""
    Nin = int(n_interior)
    r_in = np.linspace(r_min, r_max, Nin)
    dr = (r_max - r_min) / (Nin - 1)
    # Extiende a la izquierda (ghosts) y derecha por extrapolación
    left_ghosts  = r_in[0]  - dr*np.arange(ng,0,-1)
    right_ghosts = r_in[-1] + dr*np.arange(1,ng+1)
    r_full = np.concatenate([left_ghosts, r_in, right_ghosts])
    return r_full, Grid(dr), Nin

def fill_ghosts_primitives(rho, v, p, ng=NUM_GHOSTS):
    """Paridades correctas en r≈0: rho/p pares; v impar. Outflow en borde derecho."""
    N = len(rho)
    # lado izquierdo (centro)
    for i in range(ng):
        mir = 2*ng - 1 - i
        rho[i] = rho[mir]     # par
        p[i]   = p[mir]       # par
        v[i]   = -v[mir]      # impar
    # lado derecho (outflow/zero-gradient)
    last = N - ng - 1
    for k in range(1, ng+1):
        idx = last + k
        rho[idx] = rho[last]
        p[idx]   = p[last]
        v[idx]   = v[last]
    return rho, v, p

def to_conserved(rho0, v, p, eos):
    """Convert primitives to conservatives (from test.py)."""
    eps = eos.eps_from_rho_p(rho0, p)
    h   = 1.0 + eps + p/np.maximum(rho0, 1e-300)
    W   = 1.0/np.sqrt(np.maximum(1.0 - v*v, 1e-16))
    D   = rho0 * W
    Sr  = rho0 * h * W*W * v
    tau = rho0 * h * W*W - p - D
    return D, Sr, tau

def to_primitives(D, Sr, tau, eos, p_guess=None):
    """Convert conservatives to primitives using cons2prim, with optional pressure guess."""
    res = cons_to_prim(
        (D, Sr, tau), eos,
        metric=(np.ones_like(D), np.zeros_like(D), np.ones_like(D)),
        p_guess=p_guess,
    )
    return res['rho0'], res['vr'], res['p']

def max_signal_speed(rho0, v, p, eos, cfl_guard=1e-6):
    """Calculate maximum signal speed for CFL condition."""
    eps = eos.eps_from_rho_p(rho0, p)
    h   = 1.0 + eps + p/np.maximum(rho0, 1e-300)
    cs2 = np.clip(eos.gamma * p / np.maximum(rho0*h, 1e-300), 0.0, 1.0 - 1e-10)
    cs  = np.sqrt(cs2)
    return np.max(np.abs(v) + cs) + cfl_guard

# ── Gaussian pulse functions ────────────────────────────────────────────
def gaussian_pulse_1d(r, center=0.5, amplitude=0.1, width=0.05, background_rho=1e-3):
    """Create a Gaussian density pulse for advection test."""
    return background_rho + amplitude * np.exp(-((r - center) / width)**2)

def analytic_solution(r, t, center=0.5, amplitude=0.1, width=0.05,
                      velocity=0.3, background_rho=1e-3):
    """Analytic solution for advected Gaussian in flat spacetime."""
    new_center = center + velocity * t
    return gaussian_pulse_1d(r, new_center, amplitude, width, background_rho)

# ── RK3 evolution (adapted from test.py) ────────────────────────────────
def rk3_step(valencia, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve,
             dt=None, spacetime_mode="fixed_minkowski"):
    """Una etapa RK3 Shu–Osher usando compute_rhs (full approach)."""
    # Use fixed timestep if provided, otherwise CFL
    if dt is None:
        amax = max_signal_speed(rho0, v, p, eos)
        dt = 0.4 * grid.dr / amax

    # Dummy BSSN (no usado en Minkowski, pero la firma lo pide)
    bssn_vars = _DummyBSSNVars(len(r))
    bssn_d1   = _DummyBSSND1(len(r))
    background = None

    # Stage 1
    rhsD, rhsSr, rhsTau = valencia.compute_rhs(D, Sr, tau, rho0, v, p,
                                               W=None, h=None,
                                               r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                               background=background, spacetime_mode=spacetime_mode,
                                               eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)
    D1   = D   + dt*rhsD
    Sr1  = Sr  + dt*rhsSr
    tau1 = tau + dt*rhsTau
    rho1, v1, p1 = to_primitives(D1, Sr1, tau1, eos, p_guess=p)
    rho1, v1, p1 = fill_ghosts_primitives(rho1, v1, p1)

    # Stage 2
    rhsD, rhsSr, rhsTau = valencia.compute_rhs(D1, Sr1, tau1, rho1, v1, p1,
                                               W=None, h=None,
                                               r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                               background=background, spacetime_mode=spacetime_mode,
                                               eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)
    D2   = 0.75*D   + 0.25*(D1  + dt*rhsD)
    Sr2  = 0.75*Sr  + 0.25*(Sr1 + dt*rhsSr)
    tau2 = 0.75*tau + 0.25*(tau1+ dt*rhsTau)
    rho2, v2, p2 = to_primitives(D2, Sr2, tau2, eos, p_guess=p1)
    rho2, v2, p2 = fill_ghosts_primitives(rho2, v2, p2)

    # Stage 3
    rhsD, rhsSr, rhsTau = valencia.compute_rhs(D2, Sr2, tau2, rho2, v2, p2,
                                               W=None, h=None,
                                               r=r, bssn_vars=bssn_vars, bssn_d1=bssn_d1,
                                               background=background, spacetime_mode=spacetime_mode,
                                               eos=eos, grid=grid, reconstructor=recon, riemann_solver=rsolve)
    Dn   = (1.0/3.0)*D   + (2.0/3.0)*(D2   + dt*rhsD)
    Snn  = (1.0/3.0)*Sr  + (2.0/3.0)*(Sr2  + dt*rhsSr)
    taun = (1.0/3.0)*tau + (2.0/3.0)*(tau2 + dt*rhsTau)
    rhon, vn, pn = to_primitives(Dn, Snn, taun, eos, p_guess=p2)
    rhon, vn, pn = fill_ghosts_primitives(rhon, vn, pn)

    return dt, Dn, Snn, taun, rhon, vn, pn

def run_single_test(args):
    """Run convergence test for single grid resolution and reconstructor."""
    N, reconstructor_name = args

    print(f"[START] {reconstructor_name} N={N}")

    try:
        # Setup grid (using test.py approach)
        r, grid, Nin = build_grid(n_interior=N, r_min=r_min, r_max=r_max)
        N_total = len(r)

        # Setup physics objects
        eos = IdealGasEOS(gamma=gamma_gas)
        recon = create_reconstruction(reconstructor_name)
        rsolve = HLLRiemannSolver()
        valencia = ValenciaReferenceMetric()

        # Initial conditions - Gaussian pulse with constant velocity
        rho0 = gaussian_pulse_1d(r, center=gauss_center, amplitude=gauss_amplitude,
                                width=gauss_width, background_rho=background_rho)
        v = np.full_like(r, velocity)  # Constant advection velocity
        # For ideal gas EOS: need eps to calculate pressure P = (gamma-1)*rho*eps
        # Use a simple assumption: eps = p/(gamma-1)/rho, with p ~ rho (isothermal)
        eps = np.ones_like(rho0) * 0.1  # Small constant specific energy
        p = eos.pressure(rho0, eps)  # Pressure from EOS

        # Apply boundary conditions
        rho0, v, p = fill_ghosts_primitives(rho0, v, p)

        # Convert to conservative variables
        D, Sr, tau = to_conserved(rho0, v, p, eos)

        # Fixed timestep (like reference script: dt = 0.1 * dx^2)
        dr = grid.dr
        dt_fixed = 0.1 * dr * dr

        # Evolve using RK3 with fixed timestep
        t, steps = 0.0, 0
        while t < t_final and steps < 10000000:
            dt_step, D, Sr, tau, rho0, v, p = rk3_step(
                valencia, D, Sr, tau, rho0, v, p, r, grid, eos, recon, rsolve,
                dt=dt_fixed, spacetime_mode="fixed_minkowski"
            )
            t += dt_step
            steps += 1

        # Compare with analytic solution
        rho_analytic = analytic_solution(r, t, center=gauss_center,
                                       amplitude=gauss_amplitude, width=gauss_width,
                                       velocity=velocity, background_rho=background_rho)

        # Compute errors (focus on interior to avoid boundary effects)
        ng = NUM_GHOSTS
        inner = slice(ng, -ng)
        err = rho0[inner] - rho_analytic[inner]

        err_L1 = np.mean(np.abs(err))
        err_L2 = np.sqrt(np.mean(err**2))
        err_Linf = np.max(np.abs(err))

        print(f"[DONE] {reconstructor_name} N={N} L1={err_L1:.2e} L2={err_L2:.2e} L∞={err_Linf:.2e} t={t:.3f}")

        # Save solution data
        save_data = {
            'r': r[inner],
            'rho_numerical': rho0[inner],
            'rho_analytic': rho_analytic[inner],
            't_final': t,
            'velocity': velocity,
            'steps': steps
        }

        filename = f"convergence_data/{reconstructor_name}_N{N}_solution.npz"
        np.savez(filename, **save_data)

        return (reconstructor_name, N, err_L1, err_L2, err_Linf)

    except Exception as e:
        print(f"[ERROR] Exception in {reconstructor_name} N={N}: {e}")
        import traceback
        traceback.print_exc()
        return (reconstructor_name, N, np.nan, np.nan, np.nan)

def main():
    """Main convergence analysis routine."""
    print("Starting Engrenage hydrodynamics convergence test")
    print(f"Domain: [{r_min}, {r_max}], velocity: {velocity}, t_final: {t_final}")
    print(f"Grid resolutions: {Ns}")
    print(f"Reconstructors: {reconstructors}")

    # Prepare tasks for parallel execution
    tasks = [(N, recon) for recon in reconstructors for N in Ns]

    # Run in parallel
    with mp.Pool(processes=min(1, 1)) as pool:
        results = pool.map(run_single_test, tasks)

    # Organize results by reconstructor
    errors_by_reconstructor = {recon: [] for recon in reconstructors}

    for recon, N, e1, e2, ei in results:
        if not np.isnan(e1):  # Only store successful results
            errors_by_reconstructor[recon].append((N, e1, e2, ei))

    # Save error data
    for i, norm_name in enumerate(["L1", "L2", "Linf"]):
        with open(f"convergence_errors/errors_{norm_name}.csv", "w") as f:
            f.write("Reconstructor,N,Error\n")
            for recon in reconstructors:
                for entry in sorted(errors_by_reconstructor[recon]):
                    N = entry[0]
                    error = entry[i + 1]
                    f.write(f"{recon},{N},{error:.8e}\n")

    # Calculate convergence orders
    print("\n=== Convergence Analysis ===")
    print(f"{'Reconstructor':<12} {'N':>5} {'L1':>10} {'L2':>10} {'Linf':>10} {'p_L1':>7} {'p_L2':>7} {'p_Inf':>7}")

    convergence_table = []

    with open("convergence_errors/convergence_orders.csv", "w") as fcsv:
        fcsv.write("Reconstructor,N,L1,L2,Linf,p_L1,p_L2,p_Linf\n")

        for recon in reconstructors:
            rows = sorted(errors_by_reconstructor[recon])
            prev = None

            for i, (N, L1, L2, Linf) in enumerate(rows):
                if prev is None:
                    p1 = p2 = pInf = ""
                else:
                    h1, h2 = 1.0 / prev[0], 1.0 / N
                    p1 = (np.log(prev[1]) - np.log(L1)) / (np.log(h1) - np.log(h2))
                    p2 = (np.log(prev[2]) - np.log(L2)) / (np.log(h1) - np.log(h2))
                    pInf = (np.log(prev[3]) - np.log(Linf)) / (np.log(h1) - np.log(h2))
                    p1, p2, pInf = f"{p1:.3f}", f"{p2:.3f}", f"{pInf:.3f}"

                print(f"{recon:<12} {N:5d} {L1:10.2e} {L2:10.2e} {Linf:10.2e} {p1:>7} {p2:>7} {pInf:>7}")
                fcsv.write(f"{recon},{N},{L1:.8e},{L2:.8e},{Linf:.8e},{p1},{p2},{pInf}\n")
                convergence_table.append([recon.upper(), N, L1, L2, Linf, p1, p2, pInf])
                prev = (N, L1, L2, Linf)

    # Create convergence table plot
    table_formatted = []
    for row in convergence_table:
        recon, N, L1, L2, Linf, p1, p2, pInf = row
        table_formatted.append([
            recon, N,
            f"{L1:.2e}", f"{L2:.2e}", f"{Linf:.2e}",
            p1, p2, pInf
        ])

    df = pd.DataFrame(table_formatted,
                      columns=["Reconstructor", "N", "L1", "L2", "Linf", "p_L1", "p_L2", "p_Linf"])

    fig_table, ax = plt.subplots(figsize=(14, len(df)*0.4))
    ax.axis('off')
    table_mpl = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='center')
    table_mpl.auto_set_font_size(False)
    table_mpl.set_fontsize(10)
    table_mpl.scale(1, 1.5)
    plt.title('Engrenage Hydrodynamics Convergence Analysis')
    plt.tight_layout()
    plt.savefig("convergence_hydro_table.png", dpi=300, bbox_inches='tight')
    plt.close(fig_table)

    # Create convergence plots
    fig_plots, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    for recon in reconstructors:
        rows = sorted(errors_by_reconstructor[recon])
        if len(rows) > 1:
            Ns_plot = [row[0] for row in rows]
            L1_errors = [row[1] for row in rows]
            L2_errors = [row[2] for row in rows]
            Linf_errors = [row[3] for row in rows]

            ax1.loglog(Ns_plot, L1_errors, 'o-', label=recon, linewidth=2, markersize=6)
            ax2.loglog(Ns_plot, L2_errors, 'o-', label=recon, linewidth=2, markersize=6)
            ax3.loglog(Ns_plot, Linf_errors, 'o-', label=recon, linewidth=2, markersize=6)

    # Add reference lines
    N_ref = np.array(Ns)
    for ax, order, label in [(ax1, 1, "1st order"), (ax2, 2, "2nd order"), (ax3, 3, "3rd order")]:
        ax.loglog(N_ref, 1e-2 * (N_ref[0]/N_ref)**order, 'k--', alpha=0.5, label=f"{label} ref")

    ax1.set(xlabel='N', ylabel='L1 Error', title='L1 Norm Convergence')
    ax2.set(xlabel='N', ylabel='L2 Error', title='L2 Norm Convergence')
    ax3.set(xlabel='N', ylabel='L∞ Error', title='L∞ Norm Convergence')

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.invert_xaxis()  # Higher resolution on right

    plt.suptitle('Engrenage Hydrodynamics Convergence Test - Gaussian Advection')
    plt.tight_layout()
    plt.savefig("convergence_hydro_plots.png", dpi=200, bbox_inches='tight')
    plt.close(fig_plots)

    print(f"\n[INFO] Results saved:")
    print(f"  - convergence_errors/errors_*.csv")
    print(f"  - convergence_errors/convergence_orders.csv")
    print(f"  - convergence_hydro_table.png")
    print(f"  - convergence_hydro_plots.png")
    print(f"  - convergence_data/*_solution.npz")

if __name__ == "__main__":
    main()
