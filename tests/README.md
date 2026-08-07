# tests

The suite the gate runs. It mirrors src/gutachten/ directory for directory, so
the tests for a module sit under the name of that module.

Everything here runs with no display, no elevated rights and no network. That is
a condition on the design rather than a habit: a suite needing a display gets
skipped in containers, one needing an administrator raises a prompt on Windows,
and one needing a network fails in an air gapped laboratory in a way that looks
exactly like a real defect. This project is aimed at laboratories, which are
often the most restricted machines its users have.

Work that genuinely needs a display, a device or a network lives under harness/
instead, where the directory name says what it needs.
