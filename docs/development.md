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

There is no hosted CI: `maintain.sh` is the gate, and it runs Ruff, Pyright over `src/`,
and the parallel test suite. Run it before every commit, and keep `uv.lock` compatible
with `uv sync --frozen`.


## Fixture and data policy

[`../data/README.md`](../data/README.md) is the source of truth:

- executable provenance fixtures live under `data/fixtures/`;
- minimal controlled detector inputs live under `data/calibration/`;
- canonical provider-oracle originals and their manifests live under `data/synthid/`;
- evaluation-only ground truth lives under `data/evaluations/`;
- runtime detector assets live in the package; unregistered research candidates remain outside the shipped wheel.

Store each binary once. Point tests and manifests at its canonical path. Keep generated and cleaned outputs outside the repository and retain only reproducible public records allowed by the data policy.

Use synthetic byte blobs for unsupported format paths and deterministic generated negatives where a real negative fixture is unnecessary. Detection and removal tests must preserve their format-specific invariants.
