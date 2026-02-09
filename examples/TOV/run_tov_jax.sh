#!/bin/bash
# Run TOV JAX evolution sequentially for different resolutions.
# Runs one at a time to avoid saturating the machine.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/TOVEvolution_jax.py"

for NP in 1000 2000 4000 8000; do
    echo "============================================================"
    echo "  Running TOVEvolution_jax.py with num_points = $NP"
    echo "  Started at: $(date)"
    echo "============================================================"

    NUM_POINTS=$NP python "$SCRIPT"

    echo ""
    echo "  Finished num_points=$NP at: $(date)"
    echo ""
done

echo "All simulations complete."
