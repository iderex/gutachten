# transforms

The preprocessing steps, one module per step. Each transform is a named,
versioned operation with its parameters declared rather than baked in, so that a
run can record exactly which version of which step ran with which numbers, and
so the sensitivity milestone can sweep them. Edge dropoff removal, drag and
extractor mark masking, firing pin masking, outlier identification, levelling
and the bandpass filter all belong here. A step that cannot state its parameters
does not belong here, because it cannot be swept and every score that depends on
it is unfalsifiable.
