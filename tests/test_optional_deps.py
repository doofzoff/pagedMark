"""Tests for the shared optional-dependency availability guard."""

from __future__ import annotations

import importlib.machinery
import importlib.util

from pagedmark import optional_deps


def _fake_find_spec(specs: dict[str, importlib.machinery.ModuleSpec | None]):
    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        return specs[name]

    return find_spec


def _real_spec(name: str) -> importlib.machinery.ModuleSpec:
    spec = importlib.util.find_spec(name)
    assert spec is not None
    assert spec.loader is not None
    return spec


class TestModuleAvailable:
    def test_installed_module_is_available(self):
        assert optional_deps.module_available("json") is True

    def test_missing_module_is_not_available(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"ghost": None}))
        assert optional_deps.module_available("ghost") is False

    def test_namespace_package_remnant_is_not_available(self, monkeypatch):
        # A leftover data dir in site-packages (e.g. trustmark/models/ surviving
        # an uninstall) resolves to a namespace-package spec with loader=None;
        # the guard must not report it as installed.
        ns_spec = importlib.machinery.ModuleSpec("trustmark", loader=None, is_package=True)
        assert ns_spec.loader is None
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"trustmark": ns_spec}))
        assert optional_deps.module_available("trustmark") is False

    def test_any_namespace_member_fails_the_conjunction(self, monkeypatch):
        ns_spec = importlib.machinery.ModuleSpec("spandrel", loader=None, is_package=True)
        specs = {"spandrel": ns_spec, "torch": _real_spec("json")}
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec(specs))
        assert optional_deps.module_available("torch", "spandrel") is False

    def test_all_real_members_are_available(self):
        assert optional_deps.module_available("json", "logging") is True


class TestGuardsUseSharedHelper:
    def test_trustmark_is_available_rejects_namespace_remnant(self, monkeypatch):
        from pagedmark import trustmark_detector

        ns_spec = importlib.machinery.ModuleSpec("trustmark", loader=None, is_package=True)
        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec({"trustmark": ns_spec}))
        assert trustmark_detector.is_available() is False
