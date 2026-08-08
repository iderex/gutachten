"""The gate's headless and offline conditions, refused rather than intended.

CONTRIBUTING.md and ``tests/README.md`` both state that every automated test the
gate runs, runs with no display, no elevated rights and no network. Until this
file existed those were sentences, and a sentence refuses nothing: a test that
reached a host passed on a machine that could reach it and failed on the machine
of the next person who ran the suite, which is the failure the rule was written
against rather than one it prevented.

Two of the three are enforced from here. This file is imported before anything is
collected, so the enforcement covers import time as well as test bodies.

The offline half replaces the doors a process reaches a network through: name
resolution, connecting a socket, and binding one. Every higher level call goes
through one of them, so ``urllib.request.urlopen`` is refused by the same patch
that refuses a raw ``connect`` and does not need to be listed by name. The
refusal is a ``RuntimeError`` and deliberately not an ``OSError``, because the
standard library's own network callers catch ``OSError`` and re-raise it as
something that reads like an ordinary unreachable host - which is precisely the
diagnosis this rule exists to prevent somebody from spending an afternoon on.

The headless half pins the plotting backend for the whole run, so no test can
open a window whatever the machine outside it has configured. It is set here
rather than left to the environment because the default backend on a machine with
a desktop session is an interactive one, and a test that opens a window on a
workstation and not on a build agent is a defect that only one of the two sees.

**What this cannot do.** It refuses the doors in this process. A test that
launches a subprocess which reaches a network is not seen, and neither is a file
path that happens to be a network mount. The elevated rights half is not
enforced at all: nothing in the gate asks for a privileged operation, so there is
no door here to replace, and that third of the rule is carried by the documents
and by review rather than by this file. Recording which third is unenforced is
the point of saying so here.
"""

from __future__ import annotations

import os
import socket
from typing import Any, NoReturn

# Named once, so the failure message and the tests that prove it bites cannot
# drift apart into two spellings of the same rule.
RULE = "the gate runs headless, unelevated and offline"
WHERE_THE_RULE_IS_ARGUED = "https://github.com/iderex/gutachten/issues/12"


class NetworkAccessUnderTheGate(RuntimeError):
    """Raised where the suite reaches for a network.

    Not an ``OSError``. See the module docstring: the standard library turns
    those into a message about an unreachable host, and this one has to survive
    every wrapper between the test and the socket.
    """


def _refuse(door: str, detail: object) -> NoReturn:
    raise NetworkAccessUnderTheGate(
        f"{door} was called with {detail!r}, and {RULE}. This is a failure rather than "
        "a skip, because a skipped test reports the same green as one that ran. Work "
        "that genuinely needs a network lives under harness/, which the default run "
        f"does not collect. The rule is argued in {WHERE_THE_RULE_IS_ARGUED} and stated "
        "in CONTRIBUTING.md and tests/README.md."
    )


def _refuse_getaddrinfo(host: Any, port: Any, *rest: Any, **options: Any) -> NoReturn:
    _refuse("socket.getaddrinfo", (host, port))


def _refuse_gethostbyname(host: Any) -> NoReturn:
    _refuse("socket.gethostbyname", host)


def _refuse_gethostbyname_ex(host: Any) -> NoReturn:
    _refuse("socket.gethostbyname_ex", host)


def _refuse_create_connection(address: Any, *rest: Any, **options: Any) -> NoReturn:
    _refuse("socket.create_connection", address)


def _refuse_connect(self: Any, address: Any) -> NoReturn:
    _refuse("socket.socket.connect", address)


def _refuse_connect_ex(self: Any, address: Any) -> NoReturn:
    _refuse("socket.socket.connect_ex", address)


def _refuse_bind(self: Any, address: Any) -> NoReturn:
    _refuse("socket.socket.bind", address)


# Constructing a socket object is left alone. It reaches nothing on its own, and
# refusing the constructor would also refuse the local-only families that the
# standard library uses internally on some platforms, which would turn this
# guard into a source of failures that have nothing to do with a network.
socket.getaddrinfo = _refuse_getaddrinfo  # type: ignore[assignment]
socket.gethostbyname = _refuse_gethostbyname  # type: ignore[assignment]
socket.gethostbyname_ex = _refuse_gethostbyname_ex  # type: ignore[assignment]
socket.create_connection = _refuse_create_connection  # type: ignore[assignment]
socket.socket.connect = _refuse_connect  # type: ignore[assignment]
socket.socket.connect_ex = _refuse_connect_ex  # type: ignore[assignment]
socket.socket.bind = _refuse_bind  # type: ignore[assignment]

# Before matplotlib is imported anywhere, and unconditionally rather than as a
# default, so that an interactive backend already named in the environment does
# not survive into the run.
os.environ["MPLBACKEND"] = "Agg"
