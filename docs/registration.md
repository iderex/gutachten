# Searching for the registration, and what the search costs

Two cartridge cases fired by the same mechanism are almost never at the same
orientation, so the comparison has to find the orientation as well as the
displacement. This page says what the search does, what its settings decide, and
what one search costs in seconds on a real machine. The code is
`src/gutachten/compare/register.py`.

## What is searched, and what comes out

The subject is divided into cells once. The reference is turned to each
orientation in the range and each cell is correlated against the turned field.
What each cell keeps is the orientation and the displacement where it matched
best.

The result is per cell and stops there. Whether the cells agree with one another
is the congruent matching cells rule, which is
[#73](https://github.com/iderex/gutachten/issues/73), and reporting a single
registration for the pair here would settle that question in passing. The
disagreement between cells is the quantity the whole method turns on.

The reference is what gets turned, never the cells. A resampled cell has been
smoothed by the interpolation, and the fine structure the correlation is looking
at is exactly what smoothing removes.

## The four settings

`rotation_range_deg` is searched both ways from zero and `rotation_step_deg` is
the spacing between the orientations tried. A step too coarse steps over the
angle a genuine match sits at. A range too wide hands a non-matching pair more
orientations to find a spurious peak at, which is a route from a search setting
to a false positive rate and one of the more interesting things this project can
put a number on. The step has to divide the range, so that the range in a
manifest is the range that was reached.

`translation_limit` is the largest displacement in samples, along either axis, a
cell may be matched at. It exists because the correlation is taken over
placements where a tile lies wholly inside the other array, and without a bound
a tile cut from the edge of the subject cannot be placed at the displacement it
actually moved by. Measured on a 192 by 192 pair carrying a known displacement of
two samples down and three across, a three by three grid recovered it on the four
cells that could reach it and found a spurious peak on the other five; the same
construction with the bound in place recovered it on all nine. Both numbers are
asserted in `tests/unit/compare/test_register.py` rather than only written here.

`grid` and `minimum_valid` are the cell parameters and mean what they mean in
`src/gutachten/compare/cells.py`.

All five reach the run manifest through `register.record`.

## The cost that is the same everywhere

`Registration.correlations` is how many cell-by-angle correlations a search
evaluated:

    correlations = grid * grid * (2 * rotation_range_deg / rotation_step_deg + 1)

It is arithmetic on the settings, it is the same on every machine, and it is what
a sweep design multiplies up. It is asserted in the suite. It is deliberately not
a wall clock reading and no timing is written into a manifest, because two runs
of one configuration would then differ in a field that says nothing about either.

## The cost that is not: seconds, on one machine

    .venv/Scripts/python.exe harness/quiet-machine/registration_cost.py

    mode: reference, threads pinned to 1
    python 3.13.15 on Windows-11-10.0.26200-SP0
    processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
    numpy 2.5.1, scipy 1.18.0
    grid=6 minimum_valid=0.5 rotation_range_deg=10.0 rotation_step_deg=2.0 translation_limit=8 angles=11

    field      correlations   seconds   ms per correlation
     192x192           396      3.30                 8.34
     384x384           396     15.31                38.66
     768x768           396     63.56               160.52

The script is the same one on every platform and the documented invocation is
`uv run python harness/quiet-machine/registration_cost.py`. The line above is
what actually produced this output: the machine it ran on has no `uv` installed,
so the environment's own interpreter was called directly. The two differ in how
the interpreter is found and in nothing else.

This is one machine, one run each, in reference mode, and it is a measurement of
that machine rather than of this code. It is here so that a sweep can be costed
from a number somebody observed instead of from a guess, and a laboratory
planning one should re-run the script rather than quote these figures.

What the three rows say is that the time per correlation grows with the area of
the reference and not with the number of cells: the field quadruples from each
row to the next and the cost per correlation goes up by 4.6 and then 4.2. Each
correlation slides one tile over the whole reference through a transform of the
whole reference, so that is the shape to expect. A finer grid buys more cells at
close to the same cost each, and a larger scan is paid for by every one of them.

Costing a one-at-a-time sweep from this: one search at 768 by 768 with this
configuration is about a minute, so a design visiting a few hundred parameter
values on a few dozen pairs is machine-weeks rather than machine-hours on
hardware of this kind. That number belongs in the sensitivity design,
[#79](https://github.com/iderex/gutachten/issues/79), and this page is where it
gets its input.

## What is not measured here

The cost on Linux and macOS. The script runs on all three and has been run on
one, and no figure here should be read as covering the others.

The cost against a real scan. Everything above is a generated surface with no
missing samples, and a real scan carries a missing edge and a masked firing pin
impression, which change the overlap at every placement but not the number of
transforms taken.

Whether these settings are the right ones. They are one configuration, chosen so
the rows are comparable, and the whole point of the board is that no
configuration here is a recommendation.
