#!/usr/bin/env bash
# Pipeline runner: 03b (filter) → 04 (generate RVEs)
# Run this after 03_copula_to_dream3d_stats.py finishes.
#
# Usage:
#   ./run_pipeline.sh            # full run
#   ./run_pipeline.sh --dry-run  # preview filter only, no RVE generation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

echo "=========================================="
echo " Step 03b: Filter by grain count"
echo "=========================================="
"$PYTHON" "$SCRIPT_DIR/../grain_statistics/03b_filter_by_grain_count.py" "$@"

# If --dry-run was passed, stop here
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        echo "Dry-run mode: skipping RVE generation."
        exit 0
    fi
done

echo ""
echo "=========================================="
echo " Step 04: Generate RVEs (DREAM3D + DAMASK + PNG)"
echo "=========================================="
echo " Outputs: .dream3d, .xdmf, .vti, .png per sample"
echo " Skipping already-done samples (skip_if_done=True)"
echo "=========================================="
"$PYTHON" "$SCRIPT_DIR/04_generate_rves.py"

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo "=========================================="
