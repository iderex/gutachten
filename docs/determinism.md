# What a deterministic run means here

A run of this pipeline is deterministic. This page says what that means, what it
costs, and what it does not promise. The code that holds it is
`src/gutachten/determinism.py`.

## Why it matters more here than in most projects

The sensitivity study is the whole reason. It measures how far a score moves
when a parameter moves. If the same input and the same parameters give a
different score on two runs, the study cannot separate the parameter effect from
the noise of the implementation, and every number it produces is worth less than
it looks.

That is not a tidiness argument. A sensitivity analysis whose noise floor is
unknown will report small effects that are not there and miss small effects that
are, and it will do both while looking like a careful piece of work.

## The four things it means

Every source of randomness draws from a seed that is set explicitly and recorded
in the run manifest. No code calls an unseeded generator. The manifest refuses a
run that records no seed. Catching an unseeded call at the point it is made is
[#27](https://github.com/iderex/gutachten/issues/27) and is not done yet.

The number of threads the numerical backend uses is pinned in the reference
mode. A threaded reduction sums in a different order on a machine with a
different core count, and floating point addition is not associative, so an
unpinned run is a run whose last digits depend on the hardware it landed on.

Iteration over anything unordered is sorted before it affects a result. The
manifest sorts every mapping it emits, and the transform registry iterates in
identifier order rather than in the order modules happened to import each other.

The software version and the resolved dependency versions are recorded. A number
reproduced against a different SciPy is a different claim, and the version range
declared in `pyproject.toml` does not say which one ran.

## Two modes, and the cost that is accepted

Pinning threads makes a run slower than the machine can go. That cost is real,
it is accepted, and it is why there are two modes rather than one.

Reference mode pins the backend to one thread. It is what the gate runs and what
any published run uses.

Fast mode pins nothing and is faster. It carries a line in its own output saying
so, and a number from a fast run may not be used for anything reported. The line
is written into the manifest rather than left to be inferred from the mode name,
because a manifest is read by somebody who did not run it.

Neither mode is the one you get by leaving something out. Both are entered by
calling for them, so a run that never chose is a run that does not start.

## What is promised, and what is not

Promised: bit identical results on the same machine in the same locked
environment.

Promised: agreement within a declared tolerance across Linux, macOS and Windows.
That tolerance is a number to be measured rather than guessed, and it has not
been measured yet. Measuring it and enforcing it is
[#27](https://github.com/iderex/gutachten/issues/27). Until then this project
promises the same-machine half only, and no cross-platform figure quoted
anywhere here is a measurement.

Not promised, and not a gap to be closed later: bit identical results across
different processors or different library builds. A BLAS compiled for one
instruction set and one compiled for another take different code paths through
the same arithmetic, and nothing in this repository can make them agree to the
last bit. A project that claims otherwise is claiming something its tooling
cannot support.

## The limit on how the pin is applied

The pin is set through environment variables, before the numerical stack is
imported. The BLAS libraries read their thread count once, when they load, so a
variable set afterwards changes nothing while looking exactly like it worked.

That failure is silent, which is what makes it worth refusing rather than
documenting. `pin_threads` refuses once any of numpy, scipy, scikit-image,
scikit-learn or statsmodels is already imported, and names which one it found.
An honest fast run is better than a run labelled reference that was threaded.

The alternative is a library that reaches into a loaded BLAS and resizes its
pool at run time. That is a dependency this tree does not carry, and adding one
is its own decision with its own issue rather than something taken quietly.
