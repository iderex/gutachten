"""Proof that the headless and offline conditions bite, for the reason they name.

The guard itself is in ``tests/conftest.py``. These are the near misses rather
than the obvious case: the raw socket that somebody writes on purpose is the
easy one, and the interesting ones are the high level call that goes through
three wrappers on its way to the same socket, and the listening socket that a
future fixture would bind without thinking of it as a network at all.

Each of them asserts the failure message and not only the failure. A refusal that
does not name its rule is a refusal somebody works around, because the fastest
reading of an unexplained network error is that the network is at fault.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest

from tests.conftest import RULE, WHERE_THE_RULE_IS_ARGUED, NetworkAccessUnderTheGate

# Reserved for documentation by RFC 5737 and routed nowhere, so a run with the
# guard removed fails against a black hole rather than reaching a real host.
UNROUTABLE = "192.0.2.1"


def assert_the_message_names_the_rule(message: str) -> None:
    assert RULE in message, f"the refusal does not name the rule it enforces: {message}"
    assert WHERE_THE_RULE_IS_ARGUED in message, (
        f"the refusal does not point at where the rule was argued: {message}"
    )
    assert "harness/" in message, (
        "the refusal does not say where work that needs a network belongs, which is the "
        f"one thing the reader wants next: {message}"
    )


def test_connecting_a_socket_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Zero timeout so that removing the guard makes this test fail at once
        # instead of blocking for a connect timeout. A proof nobody waits for is
        # a proof nobody re-runs.
        sock.settimeout(0)
        with pytest.raises(NetworkAccessUnderTheGate) as refusal:
            sock.connect((UNROUTABLE, 80))
    assert_the_message_names_the_rule(str(refusal.value))


def test_resolving_a_name_is_refused() -> None:
    with pytest.raises(NetworkAccessUnderTheGate) as refusal:
        socket.getaddrinfo("scans.example.invalid", 443)
    assert_the_message_names_the_rule(str(refusal.value))


def test_binding_a_socket_is_refused() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessUnderTheGate) as refusal,
    ):
        sock.bind(("127.0.0.1", 0))
    assert_the_message_names_the_rule(str(refusal.value))


def test_a_high_level_fetch_is_refused_and_is_not_reported_as_an_unreachable_host() -> None:
    # The shape somebody actually writes when they want one file from somewhere.
    # `urllib` catches OSError and re-raises it as URLError, so a guard raising
    # an OSError would arrive at the reader as an ordinary network failure and
    # teach them to retry rather than to move the work under harness/.
    with pytest.raises(NetworkAccessUnderTheGate) as refusal:
        urllib.request.urlopen(f"http://{UNROUTABLE}/scan.x3p", timeout=0.001)
    assert_the_message_names_the_rule(str(refusal.value))
    assert not isinstance(refusal.value, urllib.error.URLError)


def test_the_suite_pins_a_non_interactive_plotting_backend() -> None:
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg", (
        f"the run resolved the plotting backend {matplotlib.get_backend()!r}, and {RULE}. "
        "The backend is pinned in tests/conftest.py so that no test can open a window on "
        "a machine that has a desktop session."
    )
    assert not matplotlib.rcParams["interactive"]


def test_the_default_run_does_not_collect_the_harness(pytestconfig: pytest.Config) -> None:
    # Read through pytest's own view of the effective configuration rather than
    # by parsing pyproject.toml, so this asserts what the run will actually do
    # and not what a file says it should.
    excluded = pytestconfig.getini("norecursedirs")

    assert "harness" in excluded, (
        f"the default run would collect harness/, and {RULE}. Work needing a network, a "
        "device or a display lives there and is excluded by configuration rather than by "
        f"a skip scattered through the suite. See {WHERE_THE_RULE_IS_ARGUED}."
    )
    # The setting replaces pytest's defaults rather than extending them, and
    # listing only `harness` once made the suite fail to collect at all the
    # moment a tool wrote a dot directory into the tree. So the defaults have to
    # still be there next to it.
    assert ".*" in excluded, (
        "norecursedirs replaces pytest's defaults rather than adding to them, and `.*` is "
        "no longer among them, so a dot directory in the tree is now collected"
    )
