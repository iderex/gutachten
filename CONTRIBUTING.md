# Contributing

## Getting to a passing test run

Everything below assumes a clean machine and nothing else open.

You need git and [uv](https://docs.astral.sh/uv/). uv installs the Python
interpreter as well, so there is nothing else to install first and no system
Python to keep out of the way. It installs into your own user directory and asks
for no administrator rights.

On Linux or macOS:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you would rather not run an installer script, `pip install --user uv` and
`pipx install uv` both work, and every command below becomes `python -m uv ...`.

Then:

```
git clone https://github.com/iderex/gutachten
cd gutachten
uv sync --locked
uv run pytest
```

`uv sync --locked` installs exactly the dependency versions in `uv.lock` and
fails if the lockfile does not match `pyproject.toml`, rather than resolving
something newer and passing. That is the point of it: a green run against an
unlocked resolution says nothing about what anybody else will install.

A passing run ends with a coverage table, the line
`Required test coverage of 90% reached`, and the test count. If it does, you are
set up and nothing further on this page is needed to get there.

## Before you push

```
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Those four strings are the ones the gate runs, character for character. A
documented command that differs from the gated one is worse than no
documentation, because it produces a green local run and a red pull request with
no visible reason.

`ruff format --check` reports what it would change and changes nothing. Run
`uv run ruff format .` to actually apply it. Formatting is never applied by a
workflow, because a workflow that pushes a formatting commit is a workflow with
write access to the branch, which is a far larger surface than the problem it
solves.

`mypy` runs over `src` only. The values passed around in this project are arrays
with units and an orientation convention attached, and the most expensive error
available here is a millimetre treated as a micrometre. Types do not catch all
of that. They catch the cheap half, and the cheap half still reaches a report.

## The gate runs headless, unelevated and offline

Every automated test the gate runs, runs with no display, no elevated rights and
no network, on Linux, macOS and Windows alike. This is a condition on the design
rather than something to retrofit, and it is held from an empty tree because
that is nearly free and recovering it later is not.

The failure it avoids is specific, and each half of it is a failure that looks
like something else.

A suite that needs a display gets skipped in containers, and a skipped suite
reports the same green as one that ran.

A suite that needs an administrator raises a consent prompt on Windows, and
answering that prompt settles only the one executable path that asked, so every
new build directory asks again.

A suite that needs a network fails on an aircraft, in an air-gapped laboratory
and on the day the remote host is down. Each of those failures looks exactly like
a real defect, and somebody spends an afternoon on it. This project is aimed at
forensic laboratories, which are often the most restricted machines its users
have.

Three things follow. Plots render through a non-interactive backend to a file
and no test opens a window. Nothing in the gate binds a socket or resolves a
name. Nothing in the gate needs a privileged operation.

Work that genuinely needs one of those is not pretended away. It lives under
`harness/`, it is excluded from the default run by configuration rather than by
a skip scattered through the suite, and its directory name says what it needs. A
directory called `integration` tells a reader nothing about why it was not run.

Where a test is skipped, the skip message says why, so that a run which covered
less than the whole suite cannot be read as one that covered it and found
nothing.

## Every change starts as an issue

An issue says what is wrong, what the evidence is, and what done means. If the
evidence is a number, it carries the command that produced it.

Every architecture decision gets an issue of its own, opened and settled before
anything depends on it. An issue recording what was built without recording why
that shape and not another is incomplete, and the reasons are the part a reader
six months later actually needs.

A change lands as a pull request. Its body carries the commands that back what it
claims, run at the commit being pushed.

## Claims, and the difference between kinds of them

Every asserted fact carries the command that produced it, run against the
reference the reader will have rather than against your working tree. Where a
claim cannot be backed by a command, it is written as a claim and not as a
measurement.

`verified`, `not measured` and `not evaluated on this route` are three different
statements and this project uses them as three different words. A result that was
not measured on some platform is written that way, and it does not become a pass
because the other platforms passed.

A passage admitting that something was not done survives editing. If anything it
gets sharper. Turning such a line into a tick is worse than what it replaced.

## Numerical assertions

Every numerical comparison in the suite goes through `assert_close` in
`tests/support/tolerance.py`, and the tolerance is an argument with no default. A
check in `tests/unit/` refuses a comparison made any other way, so this is a rule
rather than an intention.

A tolerance buried as a constant inside a test is chosen once to make that test
pass and is never revisited, and from then on it quietly absorbs drift. A failure
message without the tolerance in it cannot tell a rounding error from a factor of
a thousand, which in a pipeline whose defects are mostly numbers that changed is
the whole diagnosis.

## A result that makes the method look worse

That is the expected kind of result here, and it comes out of the pipeline in the
same shape as a flattering one. There is no separate path for it and no softer
wording. If you find one, report it the way you would report any other.

## Sign-off

Every commit carries a `Signed-off-by` trailer matching its author, which
`git commit -s` adds. A check refuses a commit without one.

What that trailer asserts is that you may contribute the change under the
project's license. **This repository has no license file yet**, so the trailer
currently asserts something with no referent, and that is a known gap rather than
an oversight. Which license, and whether a sign-off or a contributor agreement is
the right instrument, are both open questions for the maintainer and are
collected in [#2](https://github.com/iderex/gutachten/issues/2). Nothing here
chooses an answer to either.

Until that is settled, default copyright applies to this repository. That is
worth knowing before you spend an evening on a change.

## Reporting a security problem

See [SECURITY.md](SECURITY.md). In short: the code that parses a container from
an untrusted source is the attack surface, and a wrong analytical result is a
serious defect but is not a vulnerability report.
