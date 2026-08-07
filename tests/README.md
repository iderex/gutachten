# tests

The suite the gate runs.

Three kinds of test live here, separated by directory rather than by a naming
convention, because a convention is a thing a reader has to know and a directory
is a thing they can see.

`unit/` holds tests over single functions, and it mirrors `src/gutachten/`
directory for directory, so the tests for a module sit under the name of that
module.

`property/` holds tests over generated inputs. They ask whether something holds
for the cases nobody thought of, which is the class of input a numerical
pipeline actually meets.

`golden/` holds tests that compare a whole run against a recorded output. This
layer matters more here than in ordinary software, because most defects in a
numerical pipeline are not crashes. They are a number that quietly changed, and
the run that produced it exited zero.

`support/` is not a test directory. It holds `tolerance.py`, which is the one
place a numerical comparison is made: every numerical assertion in the suite
goes through `assert_close`, and the tolerance is an argument with no default. A
tolerance buried as a constant inside a test is a tolerance nobody revisits when
it starts hiding a real drift. A check in `unit/` refuses a comparison made any
other way.

Everything here runs with no display, no elevated rights and no network. That is
a condition on the design rather than a habit: a suite needing a display gets
skipped in containers, one needing an administrator raises a prompt on Windows,
and one needing a network fails in an air gapped laboratory in a way that looks
exactly like a real defect. This project is aimed at laboratories, which are
often the most restricted machines its users have.

Work that genuinely needs a display, a device or a network lives under harness/
instead, where the directory name says what it needs.
