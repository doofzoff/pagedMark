"""Minimal plain-text stand-ins for the rich Console/Table API.

rich was dropped as a project dependency (see the CLI plain-text refactor), but
the analysis scripts still printed through it. These shims keep the scripts
runnable without rich: ``[bold]``/``[/]``-style markup is stripped and tables
render as aligned plain text. Output goes through ``click.echo`` to match the
package CLI (no bare ``print`` in tooling).
"""

from __future__ import annotations

import re
from typing import Any

import click

# Matches rich style tags: the bare close ``[/]`` and named open/close tags such
# as ``[yellow]``, ``[bold yellow]``, ``[/green]``. Anchored to lowercase-letter
# starts so numeric/data brackets (``[1024]``, ``[file.png]``) are left intact.
_MARKUP = re.compile(r"\[(?:/|/?[a-z][a-z ]*)\]")


def _strip(obj: Any) -> str:
    return _MARKUP.sub("", str(obj))


class Table:
    """Drop-in for ``rich.table.Table`` covering add_column/add_row + render."""

    def __init__(self, *args: Any, title: str | None = None, **kwargs: Any) -> None:
        self.title = title
        self._headers: list[str] = []
        self._rows: list[list[str]] = []

    def add_column(self, header: str = "", *args: Any, **kwargs: Any) -> None:
        self._headers.append(_strip(header))

    def add_row(self, *cells: Any) -> None:
        self._rows.append([_strip(c) for c in cells])

    def render(self) -> str:
        all_rows = ([self._headers] if self._headers else []) + self._rows
        cols = max((len(r) for r in all_rows), default=0)
        widths = [0] * cols
        for row in all_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        lines: list[str] = []
        if self.title:
            lines.append(_strip(self.title))
        if self._headers:
            lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(self._headers)))
        lines.extend("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in self._rows)
        return "\n".join(lines)


class Console:
    """Drop-in for ``rich.console.Console`` covering ``print`` (with Table)."""

    def print(self, *objects: Any, **kwargs: Any) -> None:
        click.echo(" ".join(o.render() if isinstance(o, Table) else _strip(o) for o in objects))
