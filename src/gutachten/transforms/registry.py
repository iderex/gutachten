"""The registry the manifest resolves against.

A registry maps an identifier to the transform that answers to it. Three
separate obligations in this project read it rather than reading the directory:
the manifest resolver turns a recorded chain back into steps, the sweep
enumerates what there is to vary, and the check on hidden constants inspects
what is registered. So the registry being complete is load bearing, and a
transform that exists but is not in it is a defect rather than an oversight.

Registration is where a parameter record is refused. Doing it here rather than
at the first run means the failure lands on whoever added the field, at import,
with the field named, instead of on whoever was running a sweep months later.

The ordering constraints between transforms and the pipeline that runs a chain
are #49 and are not here.
"""

from __future__ import annotations

from collections.abc import Iterator

from gutachten.transforms.base import Transform, check_parameters

__all__ = ["REGISTRY", "Registry"]


class Registry:
    """Identifier to transform, with registration refusing what it should.

    Iteration is in identifier order rather than registration order. A manifest
    or a sweep that enumerates the registry writes its output in that order, and
    registration order depends on which module imported which first, which is a
    run-to-run difference nobody chose.
    """

    def __init__(self) -> None:
        self._by_identifier: dict[str, Transform] = {}

    def register(self, transform: Transform) -> Transform:
        """Add ``transform``, refusing it if it cannot be recorded properly.

        Returns the transform, so it can be used as a decorator on a class that
        is instantiated at module level, or called plainly.
        """
        identifier = transform.identifier
        if not identifier:
            raise ValueError(f"{type(transform).__name__} declares no identifier")
        if not transform.version:
            raise ValueError(
                f"transform {identifier!r} declares no version. The version is what "
                "distinguishes a result from this step today from a result recorded "
                "before its behaviour changed."
            )
        if identifier in self._by_identifier:
            existing = type(self._by_identifier[identifier]).__name__
            raise ValueError(
                f"identifier {identifier!r} is already registered, by {existing}. Two "
                "steps under one identifier make a manifest ambiguous about which one "
                "ran."
            )
        check_parameters(transform.parameters_type)
        self._by_identifier[identifier] = transform
        return transform

    def __getitem__(self, identifier: str) -> Transform:
        try:
            return self._by_identifier[identifier]
        except KeyError:
            known = ", ".join(sorted(self._by_identifier)) or "nothing"
            raise KeyError(
                f"no transform is registered as {identifier!r}; registered: {known}"
            ) from None

    def __contains__(self, identifier: object) -> bool:
        return identifier in self._by_identifier

    def __len__(self) -> int:
        return len(self._by_identifier)

    def __iter__(self) -> Iterator[Transform]:
        for identifier in sorted(self._by_identifier):
            yield self._by_identifier[identifier]

    def identifiers(self) -> tuple[str, ...]:
        """Every registered identifier, sorted."""
        return tuple(sorted(self._by_identifier))


#: The registry this project's own steps register into. A separate instance can
#: be built in a test, so a test never has to add to or remove from this one.
REGISTRY = Registry()
