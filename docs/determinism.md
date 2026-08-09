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
run that records no seed, and `refuse_unseeded_draws` in
`src/gutachten/determinism.py` refuses a draw from the global generator at the
call, naming the function. The whole registered chain runs inside that refusal
in the suite. What it cannot see is a generator built elsewhere and passed in,
or a draw made in a subprocess.

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
environment. Two runs of the chain that runs every registered step are compared
byte for byte in the suite, over the heights and over the serialised manifest.

Promised: agreement within a declared tolerance across Linux, macOS and Windows.
The tolerance is one femtometre, which is `1e-9` micrometres, and it is enforced
by `tests/golden/test_cross_platform.py` on every platform in the matrix rather
than described here.

It was measured rather than chosen. The same run was made on all three platforms
in one workflow run and the largest spread between the five recorded numbers was
`6.661338147750939e-16` micrometres, which is three units in the last place of a
double at that magnitude. The three readings, the spread and the argument for
the declared number are in `tests/golden/cross_platform.json`. The tolerance is
not the observed spread: it sits about six orders of magnitude above it, so a
differently built backend reordering its additions does not red the gate, and
three orders of magnitude below one nanometre, which is the finest height an
instrument in this field resolves, so it cannot absorb a difference anybody
could measure.

What that measurement covers is three runner images with one numerical build and
one processor family each, on the day it was taken. It is a statement about what
the gate runs and not about every machine.

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
