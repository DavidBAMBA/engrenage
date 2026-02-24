#!/bin/bash
# Run TOV JAX evolution sequentially for different resolutions.
# Runs one at a time to avoid saturating the machine.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/TOVEvolution.py"

for NP in 200 400 800; do
    echo "============================================================"
    echo "  Running TOVEvolution.py (JAX backend) with num_points = $NP"
    echo "  Started at: $(date)"
    echo "============================================================"

    JAX_RUN=1 NUM_POINTS=$NP python "$SCRIPT"

    echo ""
    echo "  Finished num_points=$NP at: $(date)"
    echo ""
done

echo "All simulations complete."
