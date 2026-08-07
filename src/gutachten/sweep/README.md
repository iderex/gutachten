# sweep

The sensitivity runs. A sweep takes a pipeline, a set of parameters with
declared plausible ranges, and a design saying how to move through them, then
runs the comparison repeatedly and records every result with the parameters that
produced it. One at a time sweeps and global sampling over the joint space both
live here. This is the subpackage the project exists for: it measures how much
of a published score is the evidence and how much is the preprocessing choice.
