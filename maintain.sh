#!/usr/bin/env bash

set -euo pipefail

uv sync --all-extras
# uv-outdated / uv-secure run via uvx (isolated env), NOT `uv run`: resolving them
# inside the project env crashes and, with set -e, aborts the whole gate before
# ruff/pyright/tests (see CLAUDE.md "Test and lint").
uvx uv-outdated
# uv-secure prints its verdict but can then crash in an internal teardown with a
# NON-ZERO exit -- observed as "annotated-doc raised exception" and later "anyio raised
# exception"; both are bugs in uv-secure's OWN uvx env, not a project vulnerability. With
# set -e that teardown crash aborts the whole gate before ruff/pyright/tests. So gate on
# the VERDICT, not the exit code: capture the output, accept the run when uv-secure
# reported all-safe (even if it then crashed), but still FAIL on a real finding (no
# all-safe line) so a genuine CVE is never masked, and fail loud if it never got a
# verdict at all (so a broken run is never silently skipped).
secure_out="$(uvx uv-secure uv.lock 2>&1)" || true
printf '%s\n' "$secure_out"
if ! grep -qE "No vulnerabilities or maintenance issues detected|All dependencies appear safe" <<<"$secure_out"; then
    echo "maintain.sh: uv-secure reported a finding or failed before its verdict -- triage before committing." >&2
    exit 1
fi
uv run ruff check --fix
uv run ruff format
# Scoped to src/: a full-project pyright run OOM-crashes node on this ML-heavy
# repo (see CLAUDE.md "Test and lint"); src/ is the authoritative strict gate.
uv run pyright src/
uv run pytest -n auto
