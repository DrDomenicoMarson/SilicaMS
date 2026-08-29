#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FILL_COMMAND="${SILICAMS_FILL_COMMAND:-silicams-fill-slit}"

fill () {
"$FILL_COMMAND" \
  --guest "$SCRIPT_DIR/thymol.gro" \
  --slit "$SCRIPT_DIR/$1/$1.gro" \
  --output "$SCRIPT_DIR/$1/$1.THY.gro" \
  --slit-geometry "$SCRIPT_DIR/$1/$1.yml" \
  --surface-plane-padding -0.03 \
  --general-cutoff 0.04
}

fill msn_0_0
fill msn_9_1
fill msn_8_2
fill msn_7_3
fill msn_6_4
