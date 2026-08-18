"""A Finder Quick Action: right-click an image, clean it, get a notification.

The gap this closes is not capability but reach. A Mac user who has the tool installed
still has to open a terminal, remember a subcommand, and type a path -- for an operation
whose input they are already looking at in a Finder window.

Two decisions are load-bearing:

- **The executable path is resolved at install time and baked in.** Services launched by
  Finder inherit a minimal environment, not a login shell: ``PATH`` there is roughly
  ``/usr/bin:/bin:/usr/sbin:/sbin``, so ``pagedmark`` -- installed by uv or pip into
  ``~/.local/bin`` or a virtualenv -- is not on it. A bundle that calls the bare name
  works when tested from a terminal and fails silently from the menu.
- **Every outcome is a notification.** A run takes tens of seconds with no window, so
  silence is indistinguishable from a broken install. Start, success and failure each
  say so, and failure carries the tail of the output rather than a shrug.
"""

from __future__ import annotations

import plistlib
import shutil
import sys
from pathlib import Path
from typing import Any

# The name is the menu item, the bundle directory, and how uninstall finds it again.
SERVICE_NAME = "Clean with pagedMark"

# Automator identifies a Quick Action by these three strings. They are the format's, not
# ours: a bundle carrying anything else is ignored by the services registry without a
# diagnostic, which is why they are named here rather than spelled inline.
_WORKFLOW_TYPE = "com.apple.Automator.servicesMenu"
_INPUT_TYPE = "com.apple.Automator.fileSystemObject.image"
_OUTPUT_TYPE = "com.apple.Automator.nothing"
_SHELL_ACTION_BUNDLE = "/System/Library/Automator/Run Shell Script.action"

# Fixed rather than generated: a stable bundle reinstalls over itself instead of
# accumulating variants, and this workflow has exactly one action to identify.
_ACTION_UUID = "6D0C4A5E-1C9B-4F3E-9E7A-1F2B3C4D5E6F"
_INPUT_UUID = "1B2C3D4E-5F60-4712-8899-AABBCCDDEEFF"
_OUTPUT_UUID = "9F8E7D6C-5B4A-4392-8817-66554433221"


def _script(executable: Path) -> str:
    """The shell body of the action, with the resolved executable baked in.

    ``all`` rather than ``invisible``: the menu item promises a clean file, and a user
    who right-clicks in Finder is not choosing between stages. Output lands beside the
    source as ``<name>_clean.<ext>``, which is where the CLI puts it too.
    """
    return f"""BIN={_quote(str(executable))}
for file in "$@"; do
  name="${{file:t}}"
  /usr/bin/osascript -e "display notification \\"Cleaning $name\\" with title \\"pagedMark\\"" &
  log=$(/usr/bin/mktemp)
  if "$BIN" all "$file" >"$log" 2>&1; then
    /usr/bin/osascript -e "display notification \\"$name is clean\\" with title \\"pagedMark\\""
  else
    # Strip the characters that would end the AppleScript string early; a mangled
    # notification is still better than none, an unescaped one is no notification.
    reason=$(/usr/bin/tail -n 2 "$log" | /usr/bin/tr -d '"\\\\' | /usr/bin/tr '\\n' ' ')
    /usr/bin/osascript -e "display notification \\"$reason\\" with title \\"pagedMark\\" subtitle \\"$name failed\\""
  fi
  /bin/rm -f "$log"
done"""


def _quote(value: str) -> str:
    """Single-quote a path for the shell, the way shlex would."""
    return "'" + value.replace("'", "'\\''") + "'"


def _info_plist() -> dict[str, Any]:
    """The services registration: what the item is called and when Finder offers it."""
    return {
        "NSServices": [
            {
                "NSMenuItem": {"default": SERVICE_NAME},
                "NSMessage": "runWorkflowAsService",
                "NSRequiredContext": {"NSApplicationIdentifier": "com.apple.finder"},
                "NSSendFileTypes": ["public.image"],
            }
        ]
    }


def _document(executable: Path) -> dict[str, Any]:
    """The Automator document: one Run Shell Script action, input as arguments."""
    return {
        "AMApplicationBuild": "521",
        "AMApplicationVersion": "2.10",
        "AMDocumentVersion": "2",
        "actions": [
            {
                "action": {
                    "AMAccepts": {
                        "Container": "List",
                        "Optional": True,
                        "Types": ["com.apple.cocoa.string"],
                    },
                    "AMActionVersion": "2.0.3",
                    "AMApplication": ["Automator"],
                    "AMParameterProperties": {
                        "COMMAND_STRING": {},
                        "CheckedForUserDefaultShell": {},
                        "inputMethod": {},
                        "shell": {},
                        "source": {},
                    },
                    "AMProvides": {"Container": "List", "Types": ["com.apple.cocoa.string"]},
                    "ActionBundlePath": _SHELL_ACTION_BUNDLE,
                    "ActionName": "Run Shell Script",
                    "ActionParameters": {
                        "COMMAND_STRING": _script(executable),
                        "CheckedForUserDefaultShell": True,
                        # 1 = the selected files arrive as "$@". The default is stdin,
                        # which would hand the action one newline-joined blob instead.
                        "inputMethod": 1,
                        "shell": "/bin/zsh",
                        "source": "",
                    },
                    "BundleIdentifier": "com.apple.Automator.RunShellScript",
                    "CFBundleVersion": "2.0.3",
                    "CanShowSelectedItemsWhenRun": False,
                    "CanShowWhenRun": True,
                    "Category": ["AMCategoryUtilities"],
                    "Class Name": "RunShellScriptAction",
                    "InputUUID": _INPUT_UUID,
                    "Keywords": ["Shell", "Script", "Command", "Run", "Unix"],
                    "OutputUUID": _OUTPUT_UUID,
                    "UUID": _ACTION_UUID,
                    "UnlocalizedApplications": ["Automator"],
                    "arguments": {},
                    "isViewVisible": 1,
                    "location": "309.000000:253.000000",
                    "nibPath": f"{_SHELL_ACTION_BUNDLE}/Contents/Resources/Base.lproj/main.nib",
                },
                "isViewVisible": 1,
            }
        ],
        "connectors": {},
        "workflowMetaData": {
            "serviceInputTypeIdentifier": _INPUT_TYPE,
            "serviceOutputTypeIdentifier": _OUTPUT_TYPE,
            # 0 = hand the action the whole selection at once, so cleaning eight files is
            # one run of eight arguments rather than eight processes each loading SDXL.
            "serviceProcessesInput": 0,
            "workflowTypeIdentifier": _WORKFLOW_TYPE,
        },
    }


def build_bundle(executable: Path) -> dict[str, bytes]:
    """The bundle as relative path -> contents, so it can be inspected without writing."""
    return {
        "Contents/Info.plist": plistlib.dumps(_info_plist()),
        "Contents/document.wflow": plistlib.dumps(_document(executable)),
    }


def service_path(home: Path) -> Path:
    """Where macOS looks for per-user services."""
    return home / "Library" / "Services" / f"{SERVICE_NAME}.workflow"


def find_executable() -> Path | None:
    """The absolute ``pagedmark`` to bake in, or None when it cannot be located.

    ``sys.executable``'s directory first: it is the interpreter actually running this
    install, so it is the right answer inside a virtualenv even when a different
    ``pagedmark`` shadows it on PATH.
    """
    candidate = Path(sys.executable).parent / "pagedmark"
    if candidate.exists():
        return candidate.resolve()
    found = shutil.which("pagedmark")
    return Path(found).resolve() if found else None


def missing_runtime(executable: Path) -> list[str] | None:
    """Which extras the target executable lacks for a full ``all`` run.

    Asked at install time because the failure it prevents is otherwise invisible: the
    default install is metadata-only, so a Quick Action pointed at it installs cleanly
    and then reports a missing dependency from every right-click. The executable may
    live in a different environment than the one running this, so it is asked rather
    than inspected -- and any failure to ask returns None, because a probe must not be
    able to block an install.
    """
    import json
    import subprocess

    try:
        completed = subprocess.run(  # noqa: S603 -- the path we just resolved ourselves
            [str(executable), "doctor", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        installed = json.loads(completed.stdout)["installed"]
    except Exception:
        return None
    return [name for name, present in installed.items() if not present and _needed_for_all(name)]


def _needed_for_all(name: str) -> bool:
    """``all`` needs pixels and diffusion; video and the extra detectors are optional."""
    return name.startswith(("pixels", "diffusion"))


def install(executable: Path, home: Path) -> Path:
    """Write the bundle, replacing any previous one. Returns the bundle path."""
    target = service_path(home)
    if target.exists():
        shutil.rmtree(target)
    for relative, contents in build_bundle(executable).items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
    return target


def uninstall(home: Path) -> bool:
    """Remove the bundle. False when there was nothing to remove."""
    target = service_path(home)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True
