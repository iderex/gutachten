# The bandpass, and what it is a bandpass of

The comparison this project makes does not operate on a scan. It operates on a
band of spatial wavelengths taken out of one, and which band that is decides the
result as much as the comparison does.

## The standard, named exactly

The filter implemented in `gutachten.transforms.bandpass` is

    ISO 16610-71:2014, Geometrical product specifications (GPS) - Filtration -
    Part 71: Robust areal filters: Gaussian regression filters

Areal rather than profile. The profile filter of the same family is
ISO 16610-31, and a surface is not a profile. This number and title were read
off the standard's catalogue entry and cross-checked against a filtration guide
published by the Physikalisch-Technische Bundesanstalt, which names 16610-31 for
profiles and 16610-71 for areal data. `gutachten.transforms.bandpass.STANDARD`
holds the same string, and a test asserts that this file and that constant have
not drifted apart.

## What is implemented, and what is not

The zeroth order form: the regression is a locally weighted mean. The standard
also describes a higher degree variant that combines the Gaussian kernel with
Savitzky-Golay coefficients in order to retain the shape of a peak, and that is
**not implemented here**. A result from this software is a result from the
zeroth order filter and is not a claim about the other one.

The text of the standard is not public. What could be verified is its number,
its title, and the shape of the filter as it is described in the open
literature. The constants the standard fixes internally could not be read, so
the one that decides the result, the tuning constant of the robust reweighting,
is a parameter of the step rather than a number written into the code as though
it had been checked.

## The two cutoffs

Two wavelengths, both in the surface's own length unit, neither with a default.
The short cutoff removes measurement noise, the long cutoff removes residual
form, and the band kept is the difference between the two smoothings. A sinusoid
sitting exactly at a cutoff comes through at half its amplitude, which is what
the Gaussian weighting's constant is chosen to make true.

That specified characteristic is not quoted in this project as a number. It is
written as the formula it comes from, in
`gutachten.transforms.bandpass.transmission`, and the tests compare a measured
transmission against that derivation. On a surface of 192 by 192 samples at 2
micrometres, with cutoffs of 20 and 120 micrometres, the measured transmission
agrees with the derived one to better than one part in a billion at every
wavelength tested.

## What robustness costs, measured

The transmission characteristic above belongs to the linear filter. The robust
reweighting is not linear, so a robust run does not have it: on a pure sinusoid
the residual is the sinusoid, and the biweight pulls down its own peaks.
Measured on the surface described above, with a tuning constant of 4 and three
passes, the transmission at the short cutoff falls from 0.500000 to 0.480074.

That is a real property of the filter and not a defect, and it is what the
robustness buys elsewhere. On the same surface with one sample lifted by 200
micrometres, the linear filter leaves 7.61 micrometres of contamination in the
band around the spike and the robust filter leaves 0.021.

## Missing samples

Weighted out, never filled. The kernel is applied to the measured heights and to
the mask separately and the result is their ratio, so a masked region
contributes nothing and biases nothing.

The failure this avoids is specific and large. A constant surface of 7
micrometres with a masked block in it filters to zero everywhere, to within
6.2e-15. The same surface with the mask filled with zeros instead filters to a
step of 3.95 micrometres at the boundary of the block, which is a feature the
size of the ones the comparison is looking for, in a place where there is no
surface at all.

## What it costs to run

A scan of 741 by 419 samples at 0.645 micrometres, with cutoffs of 5 and 250
micrometres, filters in 0.6 seconds on the machine this was measured on. The
kernel is separable and its truncation radius is derived from the point at which
the weighting stops changing a float64 sum, rather than chosen.
