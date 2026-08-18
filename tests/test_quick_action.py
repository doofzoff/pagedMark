"""Tests for the Finder Quick Action bundle.

The bundle is data, not behaviour, so it is testable without macOS: build it, parse the
plists back, and assert on the fields the services registry actually reads. What cannot
be asserted here -- that Finder shows the item -- was verified by running the generated
workflow through ``automator`` against a stub executable.
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from pagedmark import quick_action


def _parsed(executable: Path) -> tuple[dict, dict]:
    bundle = quick_action.build_bundle(executable)
    return (
        plistlib.loads(bundle["Contents/Info.plist"]),
        plistlib.loads(bundle["Contents/document.wflow"]),
    )


class TestBundle:
    def test_it_registers_one_finder_menu_item_for_images(self):
        info, _ = _parsed(Path("/opt/bin/pagedmark"))

        (service,) = info["NSServices"]
        assert service["NSMenuItem"]["default"] == quick_action.SERVICE_NAME
        assert service["NSMessage"] == "runWorkflowAsService"
        assert service["NSRequiredContext"]["NSApplicationIdentifier"] == "com.apple.finder"
        assert service["NSSendFileTypes"] == ["public.image"]

    def test_the_document_declares_itself_a_quick_action(self):
        """Automator ignores a bundle with any other type identifier, silently."""
        _, document = _parsed(Path("/opt/bin/pagedmark"))

        metadata = document["workflowMetaData"]
        assert metadata["workflowTypeIdentifier"] == "com.apple.Automator.servicesMenu"
        assert metadata["serviceInputTypeIdentifier"] == "com.apple.Automator.fileSystemObject.image"

    def test_the_selection_arrives_as_arguments_not_stdin(self):
        """The default is stdin, which would hand one newline-joined blob to a loop
        written to iterate over "$@"."""
        _, document = _parsed(Path("/opt/bin/pagedmark"))

        parameters = document["actions"][0]["action"]["ActionParameters"]
        assert parameters["inputMethod"] == 1
        assert parameters["shell"] == "/bin/zsh"

    def test_the_whole_selection_is_one_run(self):
        """Eight files must not mean eight processes each loading SDXL from scratch."""
        _, document = _parsed(Path("/opt/bin/pagedmark"))
        assert document["workflowMetaData"]["serviceProcessesInput"] == 0

    def test_the_executable_is_absolute_and_baked_in(self):
        """Finder gives services a minimal PATH, so a bare name would fail silently."""
        _, document = _parsed(Path("/opt/bin/pagedmark"))

        script = document["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]
        assert "'/opt/bin/pagedmark'" in script
        assert "$BIN" in script

    def test_a_path_with_a_space_survives_quoting(self):
        _, document = _parsed(Path("/Users/a b/Library/Application Support/bin/pagedmark"))

        script = document["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]
        assert "'/Users/a b/Library/Application Support/bin/pagedmark'" in script

    def test_a_path_with_a_quote_survives_quoting(self):
        _, document = _parsed(Path("/Users/o'brien/bin/pagedmark"))

        script = document["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]
        assert """'/Users/o'\\''brien/bin/pagedmark'""" in script

    def test_every_outcome_notifies(self):
        """A run takes tens of seconds with no window: silence reads as a broken install."""
        _, document = _parsed(Path("/opt/bin/pagedmark"))

        script = document["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]
        assert script.count("osascript") == 3, "start, success and failure each report"
        assert "failed" in script


class TestInstall:
    def test_it_writes_where_macos_looks(self, tmp_path):
        target = quick_action.install(Path("/opt/bin/pagedmark"), tmp_path)

        assert target == tmp_path / "Library" / "Services" / f"{quick_action.SERVICE_NAME}.workflow"
        assert (target / "Contents" / "Info.plist").is_file()
        assert (target / "Contents" / "document.wflow").is_file()

    def test_reinstalling_replaces_rather_than_accumulates(self, tmp_path):
        quick_action.install(Path("/opt/bin/pagedmark"), tmp_path)
        target = quick_action.install(Path("/other/pagedmark"), tmp_path)

        document = plistlib.loads((target / "Contents" / "document.wflow").read_bytes())
        script = document["actions"][0]["action"]["ActionParameters"]["COMMAND_STRING"]
        assert "'/other/pagedmark'" in script
        assert "/opt/bin" not in script

    def test_uninstall_reports_whether_there_was_anything_to_remove(self, tmp_path):
        assert quick_action.uninstall(tmp_path) is False

        quick_action.install(Path("/opt/bin/pagedmark"), tmp_path)
        assert quick_action.uninstall(tmp_path) is True
        assert not quick_action.service_path(tmp_path).exists()


class TestFindExecutable:
    def test_the_running_interpreter_wins_over_path(self, tmp_path, monkeypatch):
        """Inside a virtualenv that is the right answer even when another pagedmark
        shadows it on PATH."""
        beside = tmp_path / "bin"
        beside.mkdir()
        (beside / "pagedmark").touch()
        monkeypatch.setattr(sys, "executable", str(beside / "python"))
        monkeypatch.setattr(quick_action.shutil, "which", lambda _name: "/usr/local/bin/pagedmark")

        assert quick_action.find_executable() == (beside / "pagedmark").resolve()

    def test_it_falls_back_to_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
        found = tmp_path / "found"
        found.touch()
        monkeypatch.setattr(quick_action.shutil, "which", lambda _name: str(found))

        assert quick_action.find_executable() == found.resolve()

    def test_nothing_found_is_reported_rather_than_guessed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "executable", str(tmp_path / "nowhere" / "python"))
        monkeypatch.setattr(quick_action.shutil, "which", lambda _name: None)

        assert quick_action.find_executable() is None


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_other_platforms_are_told_rather_than_left_a_dead_bundle(system, monkeypatch, tmp_path):
    from click.testing import CliRunner

    from pagedmark.cli import main

    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    result = CliRunner().invoke(main, ["quick-action"])

    assert result.exit_code == 1
    assert "macOS" in result.output
    assert not (tmp_path / "Library").exists()


class TestQuickActionCommand:
    """The command around the bundle: what it installs, and what it refuses to."""

    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner

        return CliRunner()

    @staticmethod
    def _on_macos(monkeypatch, tmp_path):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))

    def test_it_installs_and_says_what_will_run(self, runner, tmp_path, monkeypatch):
        from pagedmark.cli import main

        self._on_macos(monkeypatch, tmp_path)
        executable = tmp_path / "pagedmark"
        executable.touch()
        monkeypatch.setattr(quick_action, "find_executable", lambda: executable)

        result = runner.invoke(main, ["quick-action"])

        assert result.exit_code == 0, result.output
        assert quick_action.service_path(tmp_path).exists()
        assert str(executable) in result.output

    def test_a_missing_executable_is_refused_not_guessed(self, runner, tmp_path, monkeypatch):
        """A bundle calling a bare name installs cleanly and then does nothing at all."""
        from pagedmark.cli import main

        self._on_macos(monkeypatch, tmp_path)
        monkeypatch.setattr(quick_action, "find_executable", lambda: None)

        result = runner.invoke(main, ["quick-action"])

        assert result.exit_code == 1
        assert "Could not find" in result.output
        assert not quick_action.service_path(tmp_path).exists()

    def test_uninstall_removes_it(self, runner, tmp_path, monkeypatch):
        from pagedmark.cli import main

        self._on_macos(monkeypatch, tmp_path)
        quick_action.install(tmp_path / "pagedmark", tmp_path)

        result = runner.invoke(main, ["quick-action", "--uninstall"])

        assert result.exit_code == 0, result.output
        assert not quick_action.service_path(tmp_path).exists()

    def test_uninstalling_what_was_never_installed_is_not_an_error(self, runner, tmp_path, monkeypatch):
        from pagedmark.cli import main

        self._on_macos(monkeypatch, tmp_path)

        result = runner.invoke(main, ["quick-action", "--uninstall"])

        assert result.exit_code == 0, result.output
        assert "Nothing to remove" in result.output
