#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 check_scope.py
lake env lean --version
lake build
lake env lean -DwarningAsError=true -DautoImplicit=false LearnedTGAxiomAudit.lean
