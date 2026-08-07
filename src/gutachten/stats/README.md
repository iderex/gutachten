# stats

The weight of evidence layer. Matching and non-matching score distributions are
estimated here, a score based likelihood ratio is computed from them with an
interval that carries its uncertainty, and calibration is reported alongside
discrimination. Nothing in this subpackage emits a match, an identification or
an exclusion, and that is a property the suite is meant to enforce rather than a
convention: a likelihood ratio with an interval is what this project produces,
and a verdict is what it refuses to produce.
