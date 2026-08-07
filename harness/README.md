# harness

What the gate does not run, kept apart from the suite rather than skipped inside
it.

Two requirements are known already. The code path that fetches from the public
scan database needs a network, and any future path that talks to a measuring
instrument needs a device. Each of those gets a directory whose name states the
requirement, because a directory called integration tells a reader nothing about
why it was not run.

Nothing here is a substitute for the suite, and nothing in the suite depends on
anything here having run.
