# The sensitivity analysis, designed and written down before it runs

This page and [sensitivity-design.json](sensitivity-design.json) are the design
of the global sensitivity analysis, fixed before the first full sweep executes.
Which parameters move, over which ranges, how the joint space is sampled, how
large the sample is and what will be reported are all here, and the commit that
carries them is the record that they were chosen before any result was seen.

The reason is plain. This project's value depends on being believable when the
result is inconvenient, and a design fixed after the results are seen is not
believable even when it is honest. Nothing below is a result and no sweep output
exists in this repository yet:

    git ls-files | grep -icE 'results|sweep-out' ; echo "exit=$?"
    0
    exit=1

## Why the headline is not a set of one at a time sweeps

Moving one parameter at a time around a chosen operating point is the obvious
design and it is the wrong headline. The steps interact. The levelling model
decides what is left for the long cutoff to remove. The masking decides which
samples the outlier threshold sees. The cell validity threshold decides which
cells survive the masking. A one at a time sweep holds all of that at the tuned
point and reports the local slope there, which is systematically the flattest
place in the space. A study that only did that would report a stability the
method does not have, using an entirely standard method, and would be hard to
argue with.

One at a time results are still produced and reported, with what they miss
stated in the same section rather than in an appendix, which is
[#85](https://github.com/iderex/gutachten/issues/85). They are what a reader of
the existing literature will look for, and refusing to show them would be a
worse kind of opacity than showing them with their limits.

## What moves

Every parameter [ranges.json](ranges.json) declares a plausible range for, and
nothing else. Forty coordinates:

    .venv/Scripts/python.exe -c "
    import collections, json, pathlib
    d = json.loads(pathlib.Path('docs/sensitivity-design.json').read_text(encoding='utf-8'))
    r = json.loads(pathlib.Path('docs/ranges.json').read_text(encoding='utf-8'))['parameters']
    c = collections.Counter(e['mapping'] for e in d['coordinates'])
    print('coordinates:', len(d['coordinates']))
    print('declared ranges:', len(r))
    print('by mapping:', dict(sorted(c.items())))
    "
    coordinates: 40
    declared ranges: 40
    by mapping: {'controlled': 9, 'direct': 27, 'mode-split': 3, 'snapped': 1}

Run against the files at the commit this page landed on. `uv` is not on the path
of the machine this was read on, so the project virtualenv was called directly;
it is the same interpreter and the same installed environment the gated command
runs.

Holding one of the forty still would be the same move the criticism above is
about, one level up: a parameter held at its tuned value cannot be reported as
unimportant, and a report that quietly did so would be making exactly the claim
this project says nobody has earned.

| mapping | coordinates |
| --- | --- |
| `direct` | 27 |
| `controlled` | 9 |
| `mode-split` | 3 |
| `snapped` | 1 |

`tests/unit/test_sensitivity_design.py` reads this table and refuses it when it
no longer matches the file it describes.

## The space is not a hypercube, and the four mappings are how it is made one

Variance based indices are defined over coordinates that vary independently. The
parameter space here is not that: nine parameters are fixed by another parameter
rather than chosen, three decide a mode as well as a value, and one is
constrained by a second one. A design that ignored that would spend most of its
sample on cells the pipeline refuses to run.

So the sample is drawn in a forty dimensional unit hypercube and mapped
deterministically onto parameter values. The mapping is part of the design and
is declared per coordinate.

**`direct`.** The unit coordinate is mapped across the declared interval, or onto
the declared set of admissible values in the order the ranges file lists them.
Whole number parameters take the floor. Twenty seven coordinates.

**`controlled`.** The value is decided entirely by another parameter and the
drawn coordinate is not read. The plane and sphere levelling models have no
order to choose and refuse one, and the polynomial model refuses its absence:

    ValueError: the 'plane' model has no order to choose and was given order=3.
    ValueError: the 'polynomial' model names no order, and the order is what decides how much of the surface it removes.

Nine coordinates, and what controls each is in the file. The consequence for the
report is stated below.

**`mode-split`.** The parameter is nullable and its null is a mode rather than an
absence: whether the levelling fit is robust at all, whether the filter is,
whether the outlier criterion is evaluated locally or over the whole surface.
The lower half of the unit coordinate is the null and the upper half is mapped
across the declared interval, and the coordinate coupled to it takes the same
branch. Half rather than any other share, because the two modes are the thing
being compared and an unequal split would decide in advance which mode the study
mostly measures. Three coordinates, each carrying one coupled partner.

**`snapped`.** The search refuses a rotation step that does not divide the
rotation range:

    ValueError: a rotation step of 3.0 degrees does not divide a range of 10.0, so the search stops at 9.0 degrees and the manifest would record a range that was never reached.

So the drawn range is rounded to the nearest whole number of the drawn step and
held inside its declared upper bound. One coordinate. The distortion this
introduces is real and is not hidden: the marginal distribution of the rotation
range is no longer uniform over its declared interval, it is uniform over the
multiples of whatever step the same draw produced, and an index for the range is
an index over that quantised marginal.

## What a controlled coordinate means when its index is read

An index for a coordinate that is only reached on one branch is an index
conditional on that branch. `level.order` is not read at all unless the drawn
model is the polynomial one, so its first order index is the variance it
explains across the whole sample, in which it is inert for two of the three
models. That understates its effect where the polynomial model is used and it is
the honest quantity to report for a study whose model choice is itself a
parameter.

The report states this beside every index of a controlled coordinate rather than
once at the top, because an index quoted out of a table is what a reader takes
away.

## The sample size, and where it comes from

Not a round number. The estimator is the Saltelli scheme over a Sobol sequence,
which costs `N(k+2)` evaluations for first order and total order indices
together, with `k` the forty coordinates and `N` the base sample. One evaluation
runs every declared pair, so the cells to be computed are `N(k+2)` times the
number of pairs.

What a cell costs was measured rather than assumed:

    .venv/Scripts/python.exe harness/quiet-machine/sweep_cost.py

    mode: reference, threads pinned to 1
    python 3.13.15 on Windows-11-10.0.26200-SP0
    processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
    numpy 2.5.1, scipy 1.18.0

    search   field      correlations   cells   seconds   seconds per cell
    base      192x192           396       2      4.15               2.07
    base      384x384           396       2     28.87              14.43
    worst     192x192         34704       2    394.26             197.13

One machine, one run each, in reference mode. It is a measurement of that machine
rather than of this code, and a laboratory planning a sweep should re-run the
script rather than quote these figures.

The cost of a cell is not one number, because the search settings are themselves
swept. The base of this design evaluates 396 correlations and the most expensive
corner the ranges admit evaluates 34704, which is 87.6 times as many. Cost
tracks that count: predicting the worst row from the base row and the ratio gives
181.41 seconds against 197.13 measured, so the model under-predicts by 8.7 per
cent at the one point where it can be checked, and the arithmetic below carries
that factor rather than ignoring it.

What the sweep will actually pay per cell is the cost at the mean number of
correlations over the declared ranges, not at the base:

    .venv/Scripts/python.exe -c "
    import numpy as np
    rng = np.random.default_rng(20260809)
    n = 1_000_000
    grid = rng.integers(3, 13, size=n)
    step = rng.uniform(0.25, 5.0, size=n)
    span = np.minimum(np.round(rng.uniform(0.0, 30.0, size=n) / step) * step, 30.0)
    corr = grid.astype(float) ** 2 * (2.0 * span / step + 1.0)
    print('mean correlations:', round(float(corr.mean()), 1))
    print('median correlations:', round(float(np.median(corr)), 1))
    "
    mean correlations: 1285.9
    median correlations: 612.0

The mean is 3.25 times the base. The draw is the same uniform draw over the
declared ranges the design itself makes, with the snapping applied, so this is
the design's own distribution rather than a convenient one.

The budget is thirty days of one machine of the kind measured, doing this and
nothing else. That is a declared choice and not a fact about anything:

    .venv/Scripts/python.exe -c "
    mean = 14.43 * 1285.9 / 396 * 1.0867
    print('mean seconds per cell at 384x384:', round(mean, 2))
    for N in (16, 32, 64, 128, 256):
        cells = N * 42 * 16
        print(f'N={N:4d} evaluations={N*42:6d} cells={cells:7d} days={cells*mean/86400:7.1f}')
    "
    mean seconds per cell at 384x384: 50.92
    N=  16 evaluations=   672 cells=  10752 days=    6.3
    N=  32 evaluations=  1344 cells=  21504 days=   12.7
    N=  64 evaluations=  2688 cells=  43008 days=   25.3
    N= 128 evaluations=  5376 cells=  86016 days=   50.7
    N= 256 evaluations= 10752 cells= 172032 days=  101.4

So the base sample is 64, which is the largest power of two that fits. A power of
two rather than the largest integer that fits, because the balance properties a
Sobol sequence is chosen for hold at powers of two and are lost between them.

## The sample is small, and that is the finding rather than a caveat

Sixty four base samples over forty coordinates is at the bottom of what a
variance based analysis is usually run at. Total order indices are the quantity
this design exists for and they are the harder of the two to estimate, so the
honest expectation, written down before the run rather than after it, is that
some of them will come back with intervals too wide to rank parameters by.

The design commits to three things because of that.

The convergence of every index with sample size is reported, computed on nested
prefixes of the same sequence, so a reader can see whether the estimate had
settled rather than taking the final value on trust.

An index whose bootstrap interval does not separate it from the next parameter in
the ranking is reported as not separated, and the ranking says so at that point
rather than presenting an order the sample cannot support.

If the interaction terms come back with intervals spanning most of their possible
range, they are reported as not estimated at this sample size and the numbers are
not printed. That is a real possible outcome of this design and deciding it now
is cheaper than deciding it under pressure later.

There is a fourth thing this design does not do, and saying why is part of
preregistering it. A screening stage, moving all forty coordinates coarsely and
carrying only the survivors into the variance based analysis, would buy a larger
`N` on fewer coordinates for the same budget. It is not preregistered here
because the selection is then data dependent, and an index computed on the
survivors of a screen carries a selection effect that this budget cannot also
measure. Trading a known small sample for an unmeasured selection is not an
improvement in honesty, which is what the sample size is being spent on.

## What is reported

For each of the forty coordinates, the first order index and the total order
index, each with a bootstrap interval at the ninety five per cent level over one
thousand resamples, resampled by source rather than by pair. Resampling by pair
treats the many pairs from one firearm as independent, which they are not, and
produces an interval that is too narrow by a wide margin.

The gap between the two indices is the interaction effect, and it is the quantity
a one at a time design cannot produce and this project needs. It is reported with
its own interval rather than as a difference of two point values.

Every quantity is reported twice, once as an effect on the score and once as an
effect on the classification outcome, which is
[#89](https://github.com/iderex/gutachten/issues/89). A parameter that moves the
congruent cell count by a few units has done nothing if every pair stays on the
same side of the decision threshold, and a great deal if it moves pairs across
it.

## The field, the pairs, and what they limit

The design runs on generated pairs at 384 by 384 samples with a spacing of 4
micrometres, which is 1.54 millimetres across, and on sixteen pairs, eight
matching and eight non-matching.

That field is smaller than a whole primer face and the design says so rather than
implying otherwise. A field covering a 9 millimetre case head at this spacing
would be about 2250 samples on a side, which is thirty four times the area, and
the measurement above says the cost per correlation grows with the area of the
reference. A design at that size is not affordable on one machine of the kind
measured, and that is a statement about the budget rather than about the method.

Sixteen pairs is likewise small. Every interval is bootstrapped by source rather
than by pair, and with eight pairs per proposition drawn from distinct seeds
there are at most eight units to resample over. An interval built on that few is
wide, and reporting it wide is the point.

Nothing in this design speaks to real scans. The reproduction check against
public data is [#77](https://github.com/iderex/gutachten/issues/77) and what
happens on data outside the curated sets is
[#91](https://github.com/iderex/gutachten/issues/91). A sensitivity result on
generated surfaces says what the pipeline does to a known input, not what
casework does to the pipeline.

## What a refused cell does

A cell the pipeline refuses stops the run today, with the cell named. The four
mappings above exist so that the sample contains none: every drawn point maps to
a parameter set every step admits, and a refusal reaching the runner is therefore
a defect in the mapping rather than a cell to be skipped. That is the design's
answer to the question `src/gutachten/sweep/runner.py` leaves open, and it is why
the answer is not "skip and count": a skipped cell is a hole in a sample whose
whole value is that it covers the space evenly, and counting holes does not fill
them.

## What is not decided here

The one at a time report is [#85](https://github.com/iderex/gutachten/issues/85)
and the run of this design is
[#87](https://github.com/iderex/gutachten/issues/87). No sampler for this design
exists yet: `src/gutachten/sweep/design.py` offers a one at a time generator and
a full factorial one, and neither is a low discrepancy sequence.

Whether the ranges are the right ones is [ranges.md](ranges.md), which records
that sixty one of ninety three bounds are this repository's own judgement. Every
index below is an index over those intervals, and a parameter reported as
unimportant is unimportant over the interval somebody here chose.
