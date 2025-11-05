# TOV Star Evolution

This directory contains scripts for simulating Tolman-Oppenheimer-Volkoff (TOV) stars in both Cowling approximation (frozen spacetime) and full BSSN evolution.

## Directory Structure

```
TOV/
├── tov_solver.py                      # TOV equation solver (Schwarzschild coordinates)
├── tov_initial_data_interpolated.py   # Initial data generator from TOV solution
├── TOVEvolution.py                    # Cowling approximation evolution
├── TOVEvolution_bssn.py               # Full BSSN + Hydro evolution
├── utils_TOVEvolution.py              # Utilities (data management, plotting)
├── plot_TOVEvolution.py               # Video generation from saved data
├── plots/                             # Output directory for plots and videos
└── README.md                          # This file
```

## Scripts

### 1. TOV Solver (`tov_solver.py`)

Solves the TOV equations in Schwarzschild coordinates to obtain equilibrium stellar structure.

**Key features:**
- Polytropic equation of state (EOS)
- Outputs: density, pressure, mass, lapse, metric functions

### 2. Initial Data Generator (`tov_initial_data_interpolated.py`)

Creates initial data for evolution by interpolating the TOV solution onto the evolution grid.

**Key features:**
- High-order interpolation (up to 11th order)
- Converts TOV solution to BSSN variables
- Includes atmosphere treatment

### 3. Cowling Evolution (`TOVEvolution.py`)

Evolves a TOV star with frozen spacetime (Cowling approximation).

**Usage:**
```bash
cd /home/yo/repositories/engrenage/examples/TOV
python TOVEvolution.py
```

**Key features:**
- RK4 time integration with fixed timestep
- Atmosphere treatment with transition zones
- Mass conservation tracking
- Saves snapshots and evolution data to HDF5

**Configuration:**
Edit the following parameters in the `main()` function:
- `r_max`: Outer boundary radius
- `num_points`: Number of grid points
- `K`, `Gamma`: Polytropic EOS parameters
- `rho_central`: Central density
- `num_steps_total`: Number of timesteps
- `SNAPSHOT_INTERVAL`: Save full domain every N steps
- `EVOLUTION_INTERVAL`: Save time series every N steps

**Output:**
- `tov_evolution_data2/`: HDF5 files with snapshots and evolution data
- `plots/`: Diagnostic plots (density, pressure, velocity, etc.)

### 4. Full BSSN Evolution (`TOVEvolution_bssn.py`)

Evolves a TOV star with dynamical spacetime (BSSN + Hydro).

**Usage:**
```bash
cd /home/yo/repositories/engrenage/examples/TOV
python TOVEvolution_bssn.py
```

**Key features:**
- Full Einstein + Hydrodynamics
- Moving-puncture gauge conditions
- Kreiss-Oliger dissipation
- Atmosphere treatment

### 5. Utilities (`utils_TOVEvolution.py`)

Provides utilities for TOV evolution simulations.

**Key components:**
- `SimulationDataManager`: HDF5 data storage for long simulations
- Plotting functions: TOV diagnostics, evolution snapshots, etc.
- Helper functions: baryon mass computation, atmosphere treatment

### 6. Video Generator (`plot_TOVEvolution.py`)

Creates videos from saved HDF5 data.

**Usage:**
```bash
cd /home/yo/repositories/engrenage/examples/TOV
python plot_TOVEvolution.py <data_directory>

# Example:
python plot_TOVEvolution.py tov_evolution_data2
```

**Output:**
Creates 3 videos in `<data_directory>/tov_plots/`:
1. **`conservative_evolution.mp4`**: Evolution of conserved variables (D, Sr, tau) vs r
2. **`primitive_evolution.mp4`**: Evolution of primitive variables (rho0, p, vr, W) vs r
3. **`timeseries_evolution.mp4`**: Evolution of central density and L1 error vs time

**Requirements:**
- ffmpeg must be installed for video generation:
  ```bash
  sudo apt install ffmpeg  # Ubuntu/Debian
  ```

## Typical Workflow

1. **Run evolution:**
   ```bash
   cd /home/yo/repositories/engrenage/examples/TOV
   python TOVEvolution.py
   ```

2. **Generate videos from saved data:**
   ```bash
   python plot_TOVEvolution.py tov_evolution_data2
   ```

3. **Analyze results:**
   - Check plots in `plots/` directory
   - Check videos in `tov_evolution_data2/tov_plots/`
   - Load HDF5 data for custom analysis

## Data Format

### Snapshots (`tov_snapshots_*.h5`)

```
snapshots/
  step_00000000/
    @attrs: step, time
    conservatives/
      D, Sr, tau
    bssn/
      phi, a, alpha, betaR, Br, K
    primitives/
      rho0, p, eps, vr, W
  step_00000100/
    ...
grid/
  r, N, r_max
```

### Evolution Time Series (`tov_evolution_*.h5`)

Datasets:
- `step`: Timestep number
- `time`: Physical time
- `rho_central`: Central density
- `p_central`: Central pressure
- `max_rho_error`: Maximum density error
- `max_p_error`: Maximum pressure error
- `max_velocity`: Maximum velocity
- `l1_rho_error`: L1 norm of density error
- `l2_rho_error`: L2 norm of density error
- `max_D`, `max_Sr`, `max_tau`: Maximum conserved variables
- `c2p_fails`: Number of cons2prim failures

### Metadata (`tov_metadata_*.json`)

Contains:
- TOV solution parameters (M, R, C, K, Gamma, rho_central)
- Atmosphere parameters
- Simulation parameters (dt, grid size, etc.)
- Timestamp

## References

- Cowling approximation: C. W. Misner, K. S. Thorne, and J. A. Wheeler, "Gravitation" (1973)
- BSSN formulation: M. Alcubierre, "Introduction to 3+1 Numerical Relativity" (2008)
- Atmosphere treatment:   ,   codes
