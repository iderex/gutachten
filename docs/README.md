# docs

What an operator reads. Installation, the one command they type, what the output
means, and what it does not mean.

The limitations belong here in full, and they also belong in the output itself,
because a manual is not what somebody has open when they are reading a number.
This directory carries the long form and the output carries the short form, and
they do not disagree.

- [determinism.md](determinism.md): what a deterministic run means here, the two
  modes and what each costs, and what this project does not promise.
- [filtering.md](filtering.md): which part of ISO 16610 the bandpass implements
  and which part of it does not, what the two cutoffs do, and what handling the
  missing samples explicitly is worth as a number.
- [registration.md](registration.md): what the search over translation and
  rotation does, what its four settings decide, and what one search costs in
  correlations and in seconds.
- [ranges.md](ranges.md): the plausible range declared for every parameter a
  sweep can reach, where each bound came from, and how much of the set is this
  repository's own judgement rather than anybody's published value.
- [parity.md](parity.md): the quality target this repository is held to, what was
  decided about each check in it, and the list of check names this repository
  produces with which of them can hold a merge at all.
- [sensitivity-design.md](sensitivity-design.md): which parameters the global
  analysis moves, how the joint space is sampled, where the sample size comes
  from, what will be reported, and why the sample is small enough that some of
  the answers may not be reportable at all.
