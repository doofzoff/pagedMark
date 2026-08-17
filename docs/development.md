# Development

Read this reference for environment setup, dependency recovery, and fixture policy.

## Local environment

- Use `uv sync --frozen --extra dev` and add only the feature extras needed for the task.
- Do not use `uv pip install` for development tools. It can re-resolve `uv.lock` outside the compatible ML dependency set.
- A default-only sync removes every pixel and model package by design. Package imports remain light through lazy exports.
- On an unreliable connection, sync `dev` plus only the required feature extras, such as `diffusion`, and run the checks directly instead of downloading every optional learned backend.
- Run `uv` from the repository root or it may create a bare environment without the project dependencies.

The optional TrustMark decoder downloads weights into its installed package directory. After pruning that extra, a leftover weights directory can make availability checks see an empty namespace package. If Pyright reports an unknown `TrustMark` import and `find_spec("trustmark")` returns a loader-less spec, remove that regenerable remnant from the active virtual environment and resync.

## Checks

`.github/workflows/test.yml` runs Ruff, Pyright over `src/` and the suite on macOS arm64
across two Python minors, with a Linux job for the paths that carry no Metal. Locally,
`bash maintain.sh` is the same gate plus the dependency and vulnerability checks. Keep
`uv.lock` compatible with `uv sync --frozen`.

### When a job fails with no steps

A job that completes in a few seconds with no runner assigned and no steps recorded has
not run at all, and its log blob does not exist -- so the reason is invisible in the usual
places. Two causes produce exactly that signature:

- an unresolvable `uses:` reference, which is rejected before the first step;
- the account being unable to run jobs, which GitHub reports only as a check-run
  annotation.

Read the annotation rather than guessing:

```bash
gh api repos/OWNER/REPO/commits/SHA/check-runs --jq '.check_runs[].id' \
  | xargs -I{} gh api repos/OWNER/REPO/check-runs/{}/annotations --jq '.[].message'
```

That is how this repository's first failure was diagnosed: it was billing, and the action
versions bumped alongside it were merely out of date, not the cause.


## Fixture and data policy

[`../data/README.md`](../data/README.md) is the source of truth:

- executable provenance fixtures live under `data/fixtures/`;
- minimal controlled detector inputs live under `data/calibration/`;
- canonical provider-oracle originals and their manifests live under `data/synthid/`;
- evaluation-only ground truth lives under `data/evaluations/`;
- runtime detector assets live in the package; unregistered research candidates remain outside the shipped wheel.

Store each binary once. Point tests and manifests at its canonical path. Keep generated and cleaned outputs outside the repository and retain only reproducible public records allowed by the data policy.

Use synthetic byte blobs for unsupported format paths and deterministic generated negatives where a real negative fixture is unnecessary. Detection and removal tests must preserve their format-specific invariants.
