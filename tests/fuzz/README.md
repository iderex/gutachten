# fuzz

The session against the container reader. It is not part of the gate, and what
it produces is.

The reader is the one component in this project that consumes bytes from
outside it, and it stacks three parsers: a zip holding an XML document holding a
binary array. Every refusal it makes is a place where a crafted file could
instead produce a crash, an unbounded allocation, or a surface that is
structurally wrong.

## The property

For any bytes at all, reading either produces a surface `gutachten.surface` will
admit, or refuses with one of the reasons the reader declares. Any other
exception is a defect, because whoever meets it in a laboratory gets a traceback
out of a library instead of a sentence saying what was wrong with the file. The
session asserts that distinction rather than an author remembering it.

## Running it

```
GUTACHTEN_FUZZ_SECONDS=300 uv run pytest tests/fuzz -m fuzz -s
```

`GUTACHTEN_FUZZ_SECONDS` is the budget and defaults to sixty. `GUTACHTEN_FUZZ_SEED`
is where the sequence starts and is fixed by default, so two runs of the same
budget try the same inputs and a finding is reproducible from the number printed
beside it. The scheduled run sets a different seed each time, because a session
that tries the same inputs every night stops finding anything after the first.

The session prints what it tried: how many inputs, and how many of them each
refusal answered. A run that tried nothing prints that rather than passing.

## Where the schedule is

`.github/workflows/fuzz.yml`, on a cron and on a manual trigger, and nowhere
else. `tests/unit/test_the_fuzz_session_is_out_of_the_gate.py` refuses the
removal of the configuration that keeps this out of the default run, and it says
in its own docstring what it cannot see: a workflow that was disabled, or a cron
that never fires, leaves it green while the session never runs. Nothing in this
tree can read that. The runs on the repository are where it is checked.

## What a finding becomes

A fixture in `tests/unit/x3p/containers.py`, in the conformance corpus, where
every pull request runs it. The session prints the reduced input as base64 so it
can go straight there, and where the shape can be constructed instead of stored
it is, because a stored blob is a fixture nobody can regenerate or read.

`a-damaged-compressed-stream` is the first one and came out of this session. The
reader caught the archive layer's own error and an early end of stream, and did
not catch the decompressor giving up, so a container whose deflate stream was
damaged left the reader as a `zlib` traceback rather than as a refusal.

## What this is not

A coverage guided fuzzer. This session does not know which branches an input
reached, so it cannot steer towards the ones nobody has reached, and a coverage
guided one would find in an hour what this finds in a day or never. What it has
instead is a corpus of containers valid by construction and a set of damage
operators drawn from how files actually arrive broken.

A clean session is therefore weak evidence and is not reported as more. Adding a
coverage guided fuzzer would mean a new dependency with a compiler behind it,
which is a decision with its own issue rather than something to fold in here.
