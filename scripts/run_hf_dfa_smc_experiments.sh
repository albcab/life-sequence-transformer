#!/usr/bin/env bash
# Same six cohorts and offsets, with 32-particle DFA-SMC and rolling GPT-2 context.
set -euo pipefail
export DFA_SMC=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_hf_experiments.sh" "$@"
