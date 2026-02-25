#!/bin/bash
# Run TOV evolution sequentially for a series of resolutions.
# Runs one resolution at a time to avoid saturating the machine.
#
# Parameters controlled here (env vars read by TOVConfig):
#   JAX_RUN      - 1=JAX backend (default), 0=Numba
#   NUM_POINTS   - grid resolution (overridden per iteration)
#   T_FINAL      - final time
#   ENABLE_RESTART    - 1=resume from last checkpoint (default), 0=fresh start
#   ENABLE_DATA_SAVING - 1=save HDF5 data (default), 0=dry run
#   SKIP_PLOTS   - 1=skip matplotlib output, 0=generate plots (default)
#
# Parameters controlled via config.py (edit directly):
#   evolution_mode, rho_central, K, Gamma, reconstructor, riemann_solver, etc.
#
# Output goes to:
#   examples/TOV/data/tov_evolution_data_rmax<N>[_jax]/tov_star_*/
#
# Usage:
#   ./run_tov_jax.sh                    # JAX, default resolutions
#   JAX_RUN=0 ./run_tov_jax.sh          # Numba backend
#   T_FINAL=500 ./run_tov_jax.sh        # override final time
#   SKIP_PLOTS=1 ./run_tov_jax.sh       # headless / no plots

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/TOVEvolution.py"

# --- Configuration ---
JAX_RUN="${JAX_RUN:-1}"
T_FINAL="${T_FINAL:-200}"
ENABLE_RESTART="${ENABLE_RESTART:-1}"
ENABLE_DATA_SAVING="${ENABLE_DATA_SAVING:-1}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
RESOLUTIONS="${RESOLUTIONS:-200 400 800}"

echo "============================================================"
echo "  TOV Evolution sweep"
echo "  Backend : $([ "$JAX_RUN" = "1" ] && echo JAX || echo Numba)"
echo "  T_final : $T_FINAL"
echo "  Restart : $ENABLE_RESTART"
echo "  Plots   : $([ "$SKIP_PLOTS" = "1" ] && echo disabled || echo enabled)"
echo "  Resols  : $RESOLUTIONS"
echo "  Output  : $SCRIPT_DIR/data/"
echo "============================================================"
echo ""

for NP in $RESOLUTIONS; do
    echo "------------------------------------------------------------"
    echo "  N = $NP  |  started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    JAX_RUN=$JAX_RUN \
    NUM_POINTS=$NP \
    T_FINAL=$T_FINAL \
    ENABLE_RESTART=$ENABLE_RESTART \
    ENABLE_DATA_SAVING=$ENABLE_DATA_SAVING \
    SKIP_PLOTS=$SKIP_PLOTS \
        python "$SCRIPT"

    echo "  N = $NP  |  finished $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "============================================================"
echo "  All simulations complete."
echo "============================================================"
