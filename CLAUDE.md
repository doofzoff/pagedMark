# Remove AI Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI provenance watermarks.

## Scope and non-goals

The project gives users control over provenance marks on content they generated or edited themselves. It does not automatically remove stock-agency, marketplace, classifieds, tiled-preview, or other marks that protect a third party's paid or copyrighted asset.

- Add visible templates only for AI-generation labels.
- Do not add stock, agency, or classifieds marks to `watermark_registry.py`.
- Keep `erase --region` generic and user-directed; do not build an automatic stock-watermark remover on it.

Full boundary and legal context: [`docs/legal-and-safety.md`](docs/legal-and-safety.md).

## How to run

```bash
uv run pagedmark --help
bash maintain.sh
```

Run `uv` from the repository root. Command selection, options, defaults, and examples live in [`docs/cli.md`](docs/cli.md). Before changing command routing, no-signal behavior, or exit codes, read the command-line section of [`docs/module-internals.md`](docs/module-internals.md).

## Configuration

GPU and ML modules are optional. Guard their imports with `is_available()`.

Optional features and installation groups are documented in [`docs/installation.md`](docs/installation.md). Model-running paths may use availability tests, while pure helpers in ML-adjacent modules must remain unit-tested without downloads.

## Test and lint

`maintain.sh` runs dependency freshness and security checks, Ruff, Pyright scoped to `src/`, and the parallel test suite. Full-project Pyright is not the project gate because the ML dependency graph can exhaust Node memory.

Command, gate, typing, and model-test invariants auto-load from [`.claude/rules/development.md`](.claude/rules/development.md). Environment recovery, CI behavior, and fixture policy live in [`docs/development.md`](docs/development.md).


## Module architecture

[`docs/module-internals.md`](docs/module-internals.md) is the canonical per-module map, including design decisions, thresholds, calibration history, incident records, and regression guards. Read the relevant section before changing a subsystem.

Research and current constraints are routed through [`docs/index.md`](docs/index.md), especially [`docs/known-limitations.md`](docs/known-limitations.md), [`docs/supported-signals.md`](docs/supported-signals.md), [`docs/synthid.md`](docs/synthid.md), and [`docs/watermarking-landscape.md`](docs/watermarking-landscape.md).

## Data safety

Follow [`data/README.md`](data/README.md) for public fixture, calibration, oracle, and evaluation layout. Store each tracked binary once and keep generated evaluation outputs outside the repository.

## Rules and conventions

Topic-specific rules live in `.claude/rules/*.md` and are auto-loaded when matching files are touched.

| File | Covers |
|---|---|
| `development.md` | Command contracts, project gate, typing boundaries, model-adjacent tests, and the detection-path measurement rule |
