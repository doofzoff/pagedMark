# Releasing

## Claim the name first (once)

`pagedmark` is not yet taken on PyPI, and the README's install line belongs to whoever
registers it. Claim it through a **pending publisher**, which needs no upload token:

1. https://pypi.org/manage/account/publishing/
2. Fill in:
   - PyPI project name: `pagedmark`
   - Owner: `doofzoff`
   - Repository: `pagedMark`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Add the environment on GitHub: Settings → Environments → New environment → `pypi`.

That reserves the name against this repository. Nothing else can publish under it, and no
secret is stored anywhere.

## Cut a release

```bash
# 1. Version lives in two places that must agree.
#    pyproject.toml -> version, and src/pagedmark/__init__.py -> __version__
# 2. Gate.
bash maintain.sh
# 3. Tag and push. The tag is what triggers publishing.
git tag v0.1.0 && git push origin v0.1.0
```

The workflow builds the sdist and wheel, refuses to continue unless both carry `LICENSE`
and `NOTICE`, and then exchanges the run's OIDC identity for a short-lived PyPI
credential. There is no long-lived token to leak or rotate.

## After the first release

Check that the README's own instruction works from a clean machine:

```bash
uv tool install --force "pagedmark[diffusion]"
pagedmark identify some.png
```
