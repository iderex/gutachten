"""What a deterministic run means here, and the two modes that follow from it.

The sensitivity study is the reason any of this exists. It measures how far a
score moves when a parameter moves. If the same input and the same parameters
give a different score on two runs, the study cannot separate the parameter
effect from the noise of the implementation, and every number it produces is
worth less than it looks.

## The four things determinism means here

**Randomness draws from a seed that is set explicitly and recorded.** No code
calls an unseeded generator. The seed is a field of the run manifest, which
refuses a manifest without one. Catching an unseeded call at the point it is
made is #27 and is not in this module.

**The thread count of the numerical backend is pinned in the reference mode.** A
threaded reduction sums in a different order on a machine with a different core
count, and floating point addition is not associative, so an unpinned run is a
run whose last digits depend on the hardware it happened to land on.

**Iteration over anything unordered is sorted before it affects a result.** The
manifest sorts every mapping it emits and the transform registry iterates in
identifier order, both for this reason.

**The software version and the resolved dependency versions are recorded.** A
number reproduced against a different SciPy is a different claim, and the
declared range in ``pyproject.toml`` does not say which one ran.

## Two modes, and why the slow one is the default for anything reported

Pinning threads makes a run slower than the machine can go, and that cost is
real rather than notional. So there are two modes and the difference between
them is in the output rather than in somebody's memory of how they ran it.

``RunMode.REFERENCE`` pins the backend to one thread. It is what the gate runs
and what a published run uses. ``RunMode.FAST`` pins nothing, is faster, and
carries a line in its own output saying it may not be used for anything
reported. A fast run is for looking at a surface, not for a number that goes
into a document.

## What is promised, and what is not

Promised: bit identical results on the same machine in the same locked
environment, and agreement within a declared tolerance across the three
supported platforms.

Not promised: bit identical results across different processors or different
library builds. That is a stronger claim than this tooling supports honestly. A
BLAS compiled for one instruction set and one compiled for another take
different code paths through the same arithmetic, and nothing in this repository
can make them agree to the last bit. Where the cross-platform tolerance actually
sits is a number to be measured in #27 rather than guessed here, and until it is
measured this module promises the same-machine half and says so.

## How the pin is applied, and the honest limit on it

By environment variable, before the numerical stack is imported. The BLAS
libraries read their thread count once, when they are loaded, so a variable set
afterwards changes nothing while looking exactly like it did. That failure is
silent, which makes it the one worth refusing, so ``pin_threads`` refuses to run
once the numerical stack is loaded rather than setting variables nobody will
read.

The alternative is a library that reaches into a loaded BLAS and changes its
pool at run time. That is a dependency this tree does not carry, and adding one
is its own decision with its own issue rather than something this module helps
itself to.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "FAST_NOTE",
    "NUMERICAL_MODULES",
    "REFERENCE_NOTE",
    "REFERENCE_THREADS",
    "THREAD_VARIABLES",
    "DeterminismRecord",
    "RunMode",
    "fast_mode",
    "pin_threads",
    "refuse_a_late_pin",
]


class RunMode(Enum):
    """Which of the two modes a run was made in."""

    REFERENCE = "reference"
    FAST = "fast"


#: Every variable a backend in this dependency graph reads its thread count
#: from. All of them are set rather than the one belonging to whichever BLAS is
#: installed, because which BLAS a wheel carries is decided at install time and
#: is not a thing this code can see.
THREAD_VARIABLES: tuple[str, ...] = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: One. A reduction over one thread sums in the order the array is laid out in,
#: which is the same order on every machine.
REFERENCE_THREADS = 1

#: The modules whose import loads a threaded backend. Once one of these is in
#: ``sys.modules`` the pin is too late to take effect.
NUMERICAL_MODULES: tuple[str, ...] = ("numpy", "scipy", "skimage", "sklearn", "statsmodels")

REFERENCE_NOTE = (
    "Reference mode. The numerical backend is pinned to one thread, so a re-run "
    "on this machine in this locked environment reproduces this output byte for "
    "byte. Agreement with another platform holds only to the declared tolerance."
)

FAST_NOTE = (
    "Fast mode. The numerical backend is not pinned, so a re-run may not "
    "reproduce this output. This run may not be used for anything reported."
)


@dataclass(frozen=True)
class DeterminismRecord:
    """How a run was made, in the form the manifest carries it.

    ``threads`` is the pinned count, or ``None`` where nothing was pinned. The
    two are not interchangeable: a recorded ``1`` says somebody pinned it, and a
    recorded nothing says the count was whatever the machine offered, which is
    the fact a reader comparing two runs needs.
    """

    mode: RunMode
    threads: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, RunMode):
            raise TypeError(f"a run mode must be a RunMode, got {self.mode!r}")
        if self.mode is RunMode.REFERENCE and self.threads != REFERENCE_THREADS:
            raise ValueError(
                f"reference mode records {self.threads!r} threads, and the reference "
                f"mode is {REFERENCE_THREADS}. A run pinned to anything else sums its "
                "reductions in an order that depends on the machine, which is what the "
                "mode exists to remove."
            )
        if self.mode is RunMode.FAST and self.threads is not None:
            raise ValueError(
                f"fast mode records {self.threads!r} threads. Fast mode pins nothing, "
                "so a recorded count would say a pin was applied that was not."
            )

    @property
    def reportable(self) -> bool:
        """Whether a number from this run may go into anything reported."""
        return self.mode is RunMode.REFERENCE

    @property
    def note(self) -> str:
        """The sentence this run carries in its own output."""
        return REFERENCE_NOTE if self.reportable else FAST_NOTE

    def to_dict(self) -> dict[str, object]:
        """The record as plain data.

        ``note`` is written out rather than left for a reader to derive from
        ``mode``. A manifest is read by somebody who did not run it, and a mode
        name is only a warning to a reader who already knows what it costs.
        """
        return {
            "mode": self.mode.value,
            "note": self.note,
            "reportable": self.reportable,
            "threads": self.threads,
        }


def refuse_a_late_pin(loaded: Iterable[str]) -> None:
    """Refuse a thread pin that arrives after the backend read its count.

    ``loaded`` is the set of module names already imported. Passed in rather
    than read here so the refusal can be shown working from a suite that has,
    necessarily, already imported numpy.
    """
    already = sorted(set(loaded) & set(NUMERICAL_MODULES))
    if already:
        raise RuntimeError(
            f"{', '.join(already)} is already imported, so pinning the thread count now "
            "sets variables nothing will read. The backend reads its count once, when it "
            "loads. Pin before importing the numerical stack, or accept fast mode and "
            "label the output as such."
        )


def pin_threads(
    environment: MutableMapping[str, str] | None = None,
    loaded: Iterable[str] | None = None,
) -> DeterminismRecord:
    """Enter reference mode: pin every backend thread variable, record having done it.

    There is no thread count to choose. A run pinned to four threads is
    reproducible on a machine with four cores and on no other, which is the
    property the pin exists to remove, so the only pin this project offers is
    the reference one.

    ``environment`` and ``loaded`` default to this process's own. They are
    arguments so a test can drive the function without editing the environment
    the rest of the suite runs in.
    """
    refuse_a_late_pin(sys.modules if loaded is None else loaded)
    target = os.environ if environment is None else environment
    for variable in THREAD_VARIABLES:
        target[variable] = str(REFERENCE_THREADS)
    return DeterminismRecord(mode=RunMode.REFERENCE, threads=REFERENCE_THREADS)


def fast_mode() -> DeterminismRecord:
    """Enter fast mode: pin nothing, and say so in the record.

    A function rather than a constant, so that entering fast mode reads the same
    way in a caller as entering reference mode does, and neither is the one that
    happens by leaving something out.
    """
    return DeterminismRecord(mode=RunMode.FAST, threads=None)
