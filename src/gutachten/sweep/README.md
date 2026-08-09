# sweep

The sensitivity runs. A sweep takes a pipeline, a set of parameters with
declared plausible ranges, and a design saying how to move through them, then
runs the comparison repeatedly and records every result with the parameters that
produced it. One at a time sweeps and global sampling over the joint space both
live here. This is the subpackage the project exists for: it measures how much
of a published score is the evidence and how much is the preprocessing choice.

## What is here

`design.py` reads a design and refuses one that could not run as written. It
reads `docs/ranges.json` rather than carrying bounds of its own, so a parameter
with no declared range cannot be swept and a value outside a declared range is
refused when the design is read rather than a day into a run.

`runner.py` runs the cells. Each one writes its row and then its manifest, and
the manifest is the completion marker, so an interrupted run is resumed rather
than restarted and the run prints how many cells it computed and how many it
reused. Any single cell can be re-run from its own manifest, which is what makes
a results table checkable one line at a time instead of only as a whole.

## What is not here

The preregistered design, with the parameters it moves, the sample size and what
it will report, is
[#79](https://github.com/iderex/gutachten/issues/79). The low discrepancy
sampling the global analysis needs is
[#87](https://github.com/iderex/gutachten/issues/87), and neither generator here
is it. The one at a time report and its limits are
[#85](https://github.com/iderex/gutachten/issues/85), and the variance based
indices are #87.

Whether a cell whose parameters no step will run should be skipped and counted
rather than stopping the run is the open question in
[#57](https://github.com/iderex/gutachten/issues/57). It belongs to the design,
so the runner raises with the cell named and decides nothing.
