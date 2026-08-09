# harness

What the gate does not run, kept apart from the suite rather than skipped inside
it.

Three requirements are known already. The code path that fetches from the public
scan database needs a network, and any future path that talks to a measuring
instrument needs a device. Each of those gets a directory whose name states the
requirement, because a directory called integration tells a reader nothing about
why it was not run.

`quiet-machine/` is the third. What lives there measures how long something
takes, and a wall clock reading is a property of the machine rather than of the
tree: the same code is slow on a loaded build agent and fast on an idle one, and
neither run says anything about the code. A timing in the gate would therefore
fail for a reason no change caused. What the suite asserts instead is the part
that is the same everywhere, such as how many correlations a search evaluates,
and the seconds are measured here and written into `docs/` with the machine they
were measured on.

Nothing here is a substitute for the suite, and nothing in the suite depends on
anything here having run.
