"""Report whether the container really has the conditions the job claims.

The suite refuses a network call from inside the process. That is a different
statement from the environment around the process having no network, and only
the second one answers the question this job exists for. So this runs before the
suite, under the identical flags, and fails the job if any of the three
conditions is missing.

It is deliberately not a test. It is imported by nothing and collected by
nothing, because a check that lived in the suite would be subject to the same
socket patching the suite installs, and would then be reporting on the guard
instead of on the machine.

The address is 192.0.2.1, reserved for documentation by RFC 5737, and the name is
under .invalid, reserved by RFC 2606. Neither can be answered by anybody, so a
reply means the container reached something it should not have.
"""

from __future__ import annotations

import os
import socket
import sys

UNROUTABLE = ("192.0.2.1", 80)
UNRESOLVABLE = "there-is-no-such-host.invalid"


def _outbound_is_blocked(problems: list[str]) -> None:
    probe = socket.socket()
    probe.settimeout(2.0)
    try:
        probe.connect(UNROUTABLE)
    except OSError as refused:
        print(f"outbound connect refused: {refused}")
    else:
        problems.append(f"the container reached {UNROUTABLE[0]}, so outbound traffic is open")
    finally:
        probe.close()

    try:
        socket.getaddrinfo(UNRESOLVABLE, 80)
    except OSError as refused:
        print(f"name resolution refused: {refused}")
    else:
        problems.append(f"the container resolved {UNRESOLVABLE}, so a resolver answered")


def _there_is_no_display(problems: list[str]) -> None:
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        value = os.environ.get(name)
        if value:
            problems.append(f"{name} is set to {value!r}")
    if os.path.exists("/tmp/.X11-unix"):
        problems.append("an X socket directory is present at /tmp/.X11-unix")
    print("DISPLAY and WAYLAND_DISPLAY unset, no X socket directory")


def _the_user_is_unprivileged(problems: list[str]) -> None:
    uid = os.getuid()
    if uid == 0:
        problems.append("the run is root, so nothing here shows the suite passes unelevated")
    print(f"running as uid {uid}, gid {os.getgid()}")

    # Linux reports the bounding set as a hex mask. Empty means no capability can
    # be acquired, which is the strongest available statement short of a sandbox.
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            bounding = [line for line in status if line.startswith("CapBnd:")]
    except OSError:
        print("no /proc/self/status to read the capability bounding set from")
        return
    if bounding:
        mask = bounding[0].split()[1]
        print(f"capability bounding set {mask}")
        if int(mask, 16) != 0:
            problems.append(f"the capability bounding set is {mask} rather than empty")


def main() -> int:
    problems: list[str] = []
    _there_is_no_display(problems)
    _the_user_is_unprivileged(problems)
    _outbound_is_blocked(problems)
    if problems:
        print("\nthis run does not have the conditions it claims:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
