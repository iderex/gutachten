# transforms

The preprocessing steps, one module per step. Each transform is a named,
versioned operation with its parameters declared rather than baked in, so that a
run can record exactly which version of which step ran with which numbers, and
so the sensitivity milestone can sweep them. Edge dropoff removal, drag and
extractor mark masking, firing pin masking, outlier identification, levelling
and the bandpass filter all belong here. A step that cannot state its parameters
does not belong here, because it cannot be swept and every score that depends on
it is unfalsifiable.

## A parameter that is not in the record does not exist

This is the decision the rest of the project rests on, so it is written here
rather than assumed.

The published performance figures for this method sit on a preprocessing chain
that was configured by hand. What this project claims is not that those figures
are wrong; it is that nobody has shown how sensitive they are to that
configuration. Showing it means moving the parameters and watching the score, and
a sweep can only move parameters it can see.

So a step that reads a tunable constant out of its own source, or that fills in a
default when a key is missing, is a step whose contribution is invisible to every
measurement made here. A sensitivity report built on such a step understates the
sensitivity while looking thorough, and that is worse than not producing one. A
number nobody produced does not get quoted. A number produced by an apparatus
with a blind spot does.

Three rules follow from that, and each is refused by something rather than asked
for.

**A parameter record declares its fields and gives none of them a default.**
`check_parameters` in `base.py` refuses a record type carrying one, at
registration, so the failure names the field and lands on whoever added it. A
default is a value chosen once by whoever was closing a ticket, and it then
travels into every run that does not mention the field, including the sweeps
meant to be varying it.

**Defaults live in named profiles.** A profile under `profiles/` is versioned,
is named in the manifest and can be compared between two runs in one word. The
tedium of typing every parameter is real, and a profile is the answer to it
because a profile is recorded. Refusing an under-specified parameter set at the
point of use is [#53](https://github.com/iderex/gutachten/issues/53).

**A transform carries its own semantic version, separate from the software
version, and it moves when the output moves for the same input.** That is what
lets a result recorded a year ago be identified as having come from a different
levelling step, instead of being compared against a current run as though the two
were one procedure. Nothing can check that somebody bumped it. What is checked is
that a version exists and is not empty, and the rest is what review is for.

## Numbers are found by reading, not by reviewing

The rule above gets broken by convenience rather than by argument. Somebody needs
a threshold, types `0.35` where they need it, and every test passes because the
number is right for the case in front of them.

`audit.py` reads the source of every registered transform and reports numeric
literals that are not structurally forced. `0`, `1` and `2` are allowed because
they are not tunable: changing the `2` in a squared term makes it a different
formula rather than a different setting. Anything else is presumed tunable until
a line says otherwise with `# structural: <reason>`, which appears in the diff
and can be disagreed with.

What it cannot see is in its own module docstring, and the short version is that
it reads literals: a number assembled from arithmetic, read from the environment,
or imported from elsewhere goes past it. It is a floor under the mistake people
actually make.

## The manifest is the run

Not a log of it. `gutachten.manifest` holds the schema: the inputs by hash, the
profile by name and version, every step in order with its version and resolved
parameters, the seed, the software and dependency versions, and the outputs by
hash. A manifest fed back in reproduces the run, which is
[#51](https://github.com/iderex/gutachten/issues/51).

The schema carries a version of its own, which moves when the meaning of a field
changes, and never because a transform changed. A reader meeting an unknown
schema version refuses it rather than reading the fields it recognises, because a
partial read of a run record produces a re-run that is not the run.

## What is not decided here

The registry holds identifiers and refuses what cannot be recorded. The ordering
constraints between steps, and the pipeline that runs a chain, are
[#49](https://github.com/iderex/gutachten/issues/49). The steps themselves are
the issues that follow it. Nothing in this directory knows what a striation is.
