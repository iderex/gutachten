# x3p

The container format. X3P is a zip holding XML metadata and a binary height
array, and this subpackage is the only place that knows that. It reads a
container into a surface, writes one back out, verifies the checksum a container
declares, and refuses input that is malformed, truncated, or shaped to attack
the reader rather than to be read. Everything that parses bytes from outside
this project lives here, because that is the real attack surface and it is
easier to defend when it is in one place. Nothing here knows what a striation
is.
