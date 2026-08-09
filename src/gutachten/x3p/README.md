# x3p

The container format. X3P is a zip holding XML metadata and a binary height
array, and this subpackage is the only place that knows that. It reads a
container into a surface, writes one back out, verifies the checksum a container
declares, and refuses input that is malformed, truncated, or shaped to attack
the reader rather than to be read. Everything that parses bytes from outside
this project lives here, because that is the real attack surface and it is
easier to defend when it is in one place. Nothing here knows what a striation
is.

## X3P is the only way a scan enters this project

X3P, the container form of ISO 25178-72, is what the public database publishes
and what the instrument vendors write. Accepting a second entry format would
mean a second set of assumptions about units, orientation and missing data, kept
in step with the first by nothing but attention. A converter from some other
vendor format is a separate program that emits X3P, not a second door into this
one, and it can be written by somebody who never reads this repository.

The cost is that a scan in a format no tool converts cannot be analysed here at
all. That is accepted. The alternative is a reader whose behaviour on the second
format is decided by whoever needed it that afternoon.

## The reader is first party

There are readers already. A C++ library, a MATLAB interface, an R package, and
an unofficial Python module under GPL-3. Two reasons not to take one.

The license of this repository is not decided, and it is not this code's to
decide. A GPL-3 dependency in the resolved graph settles it inside a lockfile.
The maintainer holds that question in
[#2](https://github.com/iderex/gutachten/issues/2), and the shape of this
milestone is what keeps it genuinely open when it gets there.

The reader is also where untrusted bytes from an evidence file enter the
process. A zip and an XML document from an unknown source is the whole attack
surface of this project, and it is the one part worth owning outright rather
than delegating to a dependency whose threat model is somebody else's.

What it costs is that format defects become ours. That is paid down by
conformance fixtures generated in the repository, which is
[#37](https://github.com/iderex/gutachten/issues/37), and by round trip
properties over generated surfaces, which is
[#35](https://github.com/iderex/gutachten/issues/35), rather than by hoping the
format is simpler than it is.

`tests/unit/x3p/test_no_third_party_x3p_reader.py` refuses a lockfile that
resolves one of the known readers, and states in its own docstring what that
check cannot see.

## What comes out of it

A `gutachten.surface.Surface` and nothing else. The fields of that type, and why
it has them, are in the module docstring of `src/gutachten/surface.py`. Reading
a container yields heights in one internal unit, micrometres, with missing
samples as not-a-number. Nothing else in this project reads a container.

Two things the plan expected the file to declare, it does not. The format fixes
lengths as metres and carries no unit element at all, so the refusal aimed at a
missing unit would never fire and what `reader.py` refuses instead is an
increment that is absent, not a number, not finite, or not positive. And no X3P
file states an axis orientation, while a `Surface` requires one, so the reader
records `Y_DOWN` and says in its own docstring that this is an assumption. A
container this project wrote carries the orientation it was written from,
because a comment is the only place the format leaves for it.
[#45](https://github.com/iderex/gutachten/issues/45) is where both are argued
and where the implausible height range check belongs; neither is settled here.
