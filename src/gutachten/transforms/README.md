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

## A chain in an impossible order is refused rather than run

Masking after a bandpass filter is the case this exists for. The filter spreads
the masked region into its neighbourhood, so a mask applied afterwards removes a
region that has already leaked into the surface around it. Nothing crashes, the
run exits zero, and the surface is wrong in a way that looks like data. A profile
is a text file somebody edits and a sweep permutes what a profile says, so an
impossible order is not something that only arrives by misunderstanding.

Each transform declares what it `produces`, what it `requires` and what it
`refuses`, in the closed vocabulary of `SurfaceProperty`. Closed because a
constraint written against `"filterd"` never fires, and it fires nowhere in a way
nothing notices, since the chain it should have refused simply runs.

The declarations name properties rather than other transforms. A constraint
saying "not after `bandpass`" is escaped by the second filtering step somebody
adds; one saying "not after anything that has filtered the surface" is not.

`pipeline.py` carries the properties forward through a chain along with the
identifier of the step that established each, which is what lets a refusal name
both transforms. It checks the whole chain, including the parameter record types,
before the first step touches a surface, so a chain wrong in its fifth step fails
before four steps of work have produced an intermediate somebody will save.

## An implemented step that nobody registered

The registry is read by the manifest resolver, by the sweep and by the constants
audit, so a step that exists and is not registered is invisible to three
obligations at once while still running for whoever calls it directly.
`unregistered_transforms` in `audit.py` imports a package, finds the classes
satisfying the interface and reports those the registry does not hold. It reads
the interface rather than a naming convention. What it cannot see is a step
defined outside the package it is pointed at.

## What is not decided here

Writing a manifest at the end of a run and re-running from one is
[#51](https://github.com/iderex/gutachten/issues/51). Refusing an
under-specified parameter set against a profile is
[#53](https://github.com/iderex/gutachten/issues/53). The steps themselves are
the issues that follow. Nothing in this directory knows what a striation is.
