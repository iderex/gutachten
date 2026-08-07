"""The internal representation of a measured surface.

A surface is a height array plus the things that make it mean something: the
sample spacing, the units those numbers are in, the orientation convention, and
a mask marking where there is no measurement. Everything downstream of the
reader works on this type and never on a bare array, because the most expensive
error available in this project is a millimetre treated as a micrometre, and an
array on its own cannot refuse it.

Empty at this stage. The layout issue creates the file so that later work has an
unambiguous place to go, and the entry format issue is what decides the fields.
"""
