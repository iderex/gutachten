"""No third party X3P reader enters the dependency graph.

The reader for the container format is written in this repository, and the
reason is not craftsmanship. The available readers are a C++ library, a MATLAB
interface, an R package and a Python module under GPL-3, and adding a copyleft
one to the resolved graph would settle the license question in a lockfile
instead of by the maintainer, who has not settled it. It is also the single
place where bytes from an untrusted evidence file enter the process, which is
the one surface in this repository worth owning outright.

So this is a check rather than a sentence. It reads the lockfile, which is what
an install actually resolves, rather than the manifest, which only lists what
was asked for directly: a transitive reader arrives through a dependency of a
dependency and appears in exactly one of the two.

**What it cannot do.** It matches distribution names against a handful of
substrings. A reader distributed under a name that says nothing about the format
passes it, and so does a general purpose library that happens to grow an X3P
loader. It is a floor under the obvious mistake, which is somebody adding the
package that comes up first in a search, and it is not a proof that nothing in
the graph can parse the format.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

LOCKFILE = Path(__file__).resolve().parents[3] / "uv.lock"

# Substrings rather than exact names, so a fork or a rename of one of the known
# readers is caught as well. `iso5436` and `iso25178` are the standards the
# format is defined by, which is what such a package tends to be named after
# when it is not named after the format itself.
FORMAT_READER_MARKERS = ("x3p", "opengps", "iso5436", "iso25178")


def test_the_lockfile_resolves_no_third_party_reader_for_the_container_format() -> None:
    locked = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))
    names = sorted(str(package["name"]) for package in locked["package"])

    assert names, f"{LOCKFILE.name} resolved no packages at all, so this check proved nothing"

    found = [
        name for name in names if any(marker in name.lower() for marker in FORMAT_READER_MARKERS)
    ]

    assert not found, (
        "the resolved dependency graph contains what looks like a third party reader for "
        f"the container format: {found}. The reader is first party here, because a "
        "copyleft one would decide the license question of this repository inside a "
        "lockfile. If this package is genuinely something else, it needs an issue saying "
        "so before the marker list changes."
    )
