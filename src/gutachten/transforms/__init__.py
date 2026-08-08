"""See README.md in this directory for what belongs here.

Importing this package registers every step in it. That is what makes the
registry complete for anybody who imported the package at all, rather than
complete only for a caller who happened to import the right modules, and the
registry is what the manifest resolver, the sweep and the constants audit all
read. A step added to this directory and not added here is what
``unregistered_transforms`` refuses.
"""

from gutachten.transforms import bandpass, edge, level, outliers

__all__ = ["bandpass", "edge", "level", "outliers"]
