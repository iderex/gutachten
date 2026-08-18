# Security

## Reporting a vulnerability

Report privately, through GitHub's private vulnerability reporting on this
repository: open the Security tab and use "Report a vulnerability". That opens a
channel visible only to the maintainer.

The form is here, without navigating:

<https://github.com/iderex/gutachten/security/advisories/new>

Do not open a public issue for something that is exploitable. Do open a public
issue for everything else, including the analytical defects described below,
because those are the ones that benefit from being argued about in the open.

What is useful in a report: what you did, what happened, and what you expected. A
file that reproduces it is worth more than a description of one. If you have a
proof of concept, attach it rather than describing where to find it.

There is no bounty. There is one maintainer, so an acknowledgement may take a
few days.

## What the attack surface actually is

This is worth stating carefully, because for this project the honest scope is not
the obvious one.

**The container reader is the attack surface.** X3P is a zip archive holding XML
and a binary height array, and the code that opens one is parsing a file that
came from somewhere else. A laboratory receives scans from other laboratories, a
database, a defence expert, or an opposing party. Every classic failure of that
shape is in scope here: a zip that expands to far more than it claims, entries
whose paths escape the extraction directory, XML that pulls in external entities
or expands recursively, a declared array size that does not match the bytes
present, and a checksum that is present but never verified. So is anything that
turns reading a file into running code, writing outside a working directory, or
consuming memory without a bound.

Everything else the project reads from outside is in scope on the same terms: a
run manifest, a parameter profile, a cached scan, anything the acquisition path
brings in.

**The analytical results are not a security boundary.** A score, a likelihood
ratio, an interval and a calibration figure are outputs of a method. They are not
access decisions, and nothing in this project authorises, authenticates or
restricts anything. There is no privilege here to escalate.

That distinction is not a way of dismissing analytical defects. It is the
opposite. **A wrong likelihood ratio is a far worse outcome than a crash**,
because a crash is visible and a wrong number is not, and a number from this
project may end up in a report that a court reads. If you find that a
preprocessing step misbehaves on a real surface, that a score is not what the
method it names would produce, that an interval is too narrow, or that a result
is not reproducible from its manifest, that is one of the most valuable things
you can report here.

It is still not a vulnerability report, and it goes in a public issue. Routing it
through the private channel would slow it down and would keep an argument that
belongs in the open out of it.

## Versions

Nothing has been released yet, so there is no supported version to name. Until
there is, reports are taken against the current state of `main`.

## What has not been done

No security review of this repository by anyone outside the project has taken
place. There is no fuzzing of the container reader yet, and no static analysis or
code scanning beyond the workflow audit that runs on the workflows themselves.
Each of those is tracked as its own issue on the board rather than assumed. This
section is the current state and not an aspiration, and it will be shortened only
by work landing, never by rewording.
