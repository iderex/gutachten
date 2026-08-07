# profiles

Named parameter sets for the pipeline, one file per profile.

A profile fixes every preprocessing parameter so that a run can be named rather
than described, and so that two people comparing results can establish in one
word whether they ran the same thing. One profile reproduces the configuration
of the published work this project is measuring against, which is what makes the
reproduction check a comparison rather than an approximation.

A parameter that is not in a profile is a parameter somebody chose silently, and
that is the failure this directory exists against.
