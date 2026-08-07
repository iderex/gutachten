"""The run record.

A run writes down what it did: the input, every transform with its name,
version and parameters, the resolved dependency versions, and the identity of
the build that produced the output. The manifest is what makes a result
re-runnable by somebody else rather than only by the person who ran it, and it
is what the sensitivity study varies against.

Empty at this stage. The preprocessing milestone decides its shape.
"""
