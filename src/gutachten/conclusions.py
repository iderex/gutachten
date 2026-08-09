"""No output of this project states a match, an identification or an exclusion.

The project exists because a binary conclusion is being reported from a method
whose error rate is uncertain and examiner dependent. A conclusion emitted from
this software, even as a convenience field beside the ratio, would be the one
line quoted out of everything around it, and it would be quoted with this
project's name attached. So the constraint is read off the output rather than
remembered by whoever writes the next template.

Two halves, because the constraint can be broken in two unrelated ways.

## The text half

``conclusion_words`` reads a piece of output and reports every word in it that
states a conclusion. The vocabulary is stems rather than whole words, in
``REFUSED_STEMS``, so the inflections nobody predicts are caught: ``matched``,
``matching``, ``identifies``, ``excluded`` and ``concluding`` all fall out of
five stems, and a check listing whole words would have to grow every time
somebody conjugates one differently.

That vocabulary is also, unavoidably, the vocabulary of the field. Three
routes exist for a legitimate use, and each of them makes the reason visible in
the output rather than in a switch.

**A name is not a statement.** ``congruent-matching-cells`` is what the method
is called. The word ``matching`` in it modifies ``cells`` and says nothing about
the pair in front of a reader, and no reading of the surrounding text will tell
a check that. So the names are declared, in ``NAMES``, one entry per name with
the reason it is a name. The list is short on purpose and it grows by argument:
an entry added to it is a claim in a diff that a reader can disagree with, which
is what a per-file or per-line switch would not be.

``PLAIN_WORDS`` is the same idea for one word rather than a phrase, and it holds
``identifier``. That word shares nine letters with ``identified``, it is the key
every step and every method is recorded under, and it is on almost every line of
every manifest this project writes. A vocabulary refusing it would refuse the
run record itself, which is how a check gets switched off rather than fixed.

**A quotation of the literature carries its source.** The papers this project
argues with are full of these exact words, and the documents quoting them are
documents this project has to write. A line carrying ``[cited: <source>]`` is
exempt for that reason. The mark names who wrote the sentence, which is the
thing that makes it a quotation rather than a statement, so the exemption and
the attribution are one act instead of two.

**A measurement over a set carries the set.** A count of how many labelled pairs
a threshold put on which side is a property of a threshold applied to a ground
truth set, and it is what the sensitivity milestone reports. It is not a
statement about an individual comparison. A line carrying
``[measured over <set>]`` is exempt for that reason.

The measurement mark is what keeps the check off the sensitivity report, and it
is a requirement rather than a courtesy: a count reported without naming the set
it was measured over is already a defect on this board, which is what #103 is
about. So the mark asks for something the report owes anyway.

**Both marks name their referent or they do not exempt.** ``[cited:]`` and
``[measured over]`` with nothing after them are refused like an unmarked line.
An empty mark is a switch, and a switch is what this design is avoiding.

**The bound.** The check reads the mark. It cannot tell whether the source
exists, whether the sentence is really that source's, or whether the named set
was really the one measured over. What the mark buys is that the claim is in the
output where a reader meets it, next to the word it is excusing.

**The unit is the line.** A quotation running over four lines carries the mark
on four lines. That is the cost, and it is chosen against the alternative: a
mark that opened a region would sit at the top of a document and exempt
everything anybody added underneath it afterwards, which is the failure this
whole module is written against.

## The field half

``conclusion_fields`` reads a result type and reports every boolean field whose
name states a conclusion. ``is_match``, ``identified``, ``excluded``: each of
them is one attribute access away from a caller printing the thing this project
refuses to print, and none of them is caught by the text half, because a field
that nothing has rendered yet emits no text at all.

``conclusion_fields_in`` points that at a whole package, so the guard is over
the result types this project has rather than over the ones somebody remembered
to list in a test. A record added tomorrow is read the day it is added.

Booleans rather than every field, because a boolean is the shape a conclusion
takes. ``congruent`` is an integer count and is the score this project does
report; ``cells`` is the set that produced it. Neither is a verdict and neither
is refused. A field that smuggles a conclusion through as a string is not caught
here, and that is stated rather than asserted away.

Result records rather than every record. ``exclude_drag`` on a masking step is a
boolean whose name carries a refused stem and it is a setting: it asks for a
region of a surface to be left out, which is the parameter #57 exists to make
movable at all. The walk is told which records are settings by whoever calls it,
for the reason given at ``conclusion_fields_in``.

## What this does not reach

It reads the text it is handed. Nothing here walks the tree looking for output,
because what counts as output is decided by the code that renders it, and the
tests beside this module are where the artefacts this project renders today are
named and read. A rendered artefact that no test hands to this check is not
covered by it, and the day a report template lands is the day one has to.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType

__all__ = [
    "CITED",
    "MEASURED_OVER",
    "NAMES",
    "PLAIN_WORDS",
    "REFUSED_STEMS",
    "ConclusionField",
    "ConclusionWord",
    "conclusion_fields",
    "conclusion_fields_in",
    "conclusion_words",
]

#: The stems a conclusion is written with. Prefix matched against a lowercased
#: word, so every inflection of each one is covered by the one entry.
#:
#: ``inconclu`` is listed beside ``conclu`` rather than folded into it because
#: "inconclusive" does not start with "conclusive"; it is the third of the three
#: conclusions this field reports and leaving it out would admit exactly one of
#: them.
REFUSED_STEMS: tuple[str, ...] = (
    "conclu",
    "exclud",
    "exclus",
    "identif",
    "inconclu",
    "match",
)

#: Names that carry a refused stem and state nothing. One entry per name, with
#: the reason it is a name, because the list is the part of this check that can
#: be widened until nothing is refused.
NAMES: tuple[tuple[str, str], ...] = (
    (
        "high congruent matching cells",
        "the name of the variant, in prose. The longer name is listed before the "
        "shorter one so it is removed whole rather than leaving 'high' behind.",
    ),
    (
        "high-congruent-matching-cells",
        "the identifier the variant is recorded under, in a manifest and in a score record",
    ),
    (
        "congruent matching cells",
        "the name of the method, in prose. 'matching' modifies 'cells' and says "
        "nothing about a pair of items.",
    ),
    (
        "congruent-matching-cells",
        "the identifier the method is recorded under, in a manifest and in a score record",
    ),
)

#: Whole words that begin with a refused stem and state nothing. Separate from
#: ``NAMES`` because these are single words rather than phrases, and short
#: because each entry is a hole: the word is admitted wherever it appears.
#:
#: ``identifier`` is the reason this list exists. It shares nine letters with
#: ``identified``, it is the key every step and every method is recorded under,
#: and it appears in every manifest this project writes. A vocabulary that
#: refused it would refuse the run record itself on the first line that carries
#: a step.
PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    (
        "identifier",
        "the key a step, a method or a scan is recorded under. It names a thing "
        "rather than saying what a comparison showed.",
    ),
    ("identifiers", "the plural of the same"),
)

#: A line quoting the literature names its source here.
CITED = re.compile(r"\[cited:\s*[^\]\s][^\]]*\]")

#: A line reporting counts over a labelled set names the set here.
MEASURED_OVER = re.compile(r"\[measured over\s+[^\]\s][^\]]*\]")

_WORD = re.compile(r"[A-Za-z]+")
_PLAIN = frozenset(word for word, _reason in PLAIN_WORDS)


@dataclass(frozen=True)
class ConclusionWord:
    """One word of conclusion in output that neither names a source nor a set."""

    source: str
    line: int
    column: int
    word: str

    def __str__(self) -> str:
        return (
            f"{self.source}:{self.line}:{self.column} says {self.word!r}. No output of "
            "this project states a match, an identification or an exclusion (#101). "
            "Quote the literature with '[cited: <source>]' on the line, report a count "
            "over a labelled set with '[measured over <set>]' on the line, or say what "
            "the score is without saying what it means."
        )


@dataclass(frozen=True)
class ConclusionField:
    """One boolean field on a result type whose name states a conclusion."""

    record: str
    field: str

    def __str__(self) -> str:
        return (
            f"{self.record}.{self.field} is a boolean whose name states a conclusion. "
            "A result type carries the score, its denominator and the propositions, and "
            "no field a caller can print as a verdict (#101)."
        )


def _without_names(line: str) -> str:
    """``line`` with every declared name blanked out, length preserved.

    Blanked rather than deleted, so a column reported afterwards is still the
    column in the line the reader is looking at.
    """
    lowered = line.lower()
    kept = list(line)
    for name, _reason in NAMES:
        start = lowered.find(name)
        while start != -1:
            for index in range(start, start + len(name)):
                kept[index] = " "
            lowered = lowered[:start] + " " * len(name) + lowered[start + len(name) :]
            start = lowered.find(name)
    return "".join(kept)


def conclusion_words(text: str, *, source: str = "output") -> list[ConclusionWord]:
    """Every word in ``text`` that states a conclusion and is not accounted for.

    ``source`` is what the finding names, so a caller reading a file passes its
    path and a caller checking a string it just rendered passes what rendered
    it. In line and column order, which is reading order.
    """
    found: list[ConclusionWord] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if CITED.search(line) or MEASURED_OVER.search(line):
            continue
        for word in _WORD.finditer(_without_names(line)):
            lowered = word.group().lower()
            if not lowered.startswith(REFUSED_STEMS):
                continue
            if lowered in _PLAIN:
                continue
            found.append(
                ConclusionWord(
                    source=source,
                    line=number,
                    column=word.start() + 1,
                    word=word.group(),
                )
            )
    return found


def conclusion_fields(record: type) -> list[ConclusionField]:
    """Every boolean field on ``record`` whose name states a conclusion.

    Sorted by field name, so two runs over one type report the same list in the
    same order rather than the order the fields were declared in.
    """
    if not dataclasses.is_dataclass(record):
        raise TypeError(
            f"{record.__name__} is not a dataclass, so its fields cannot be read. A "
            "result type this project emits is a frozen record, and one that is not "
            "cannot be checked by reading it."
        )
    found = [
        ConclusionField(record=record.__name__, field=field.name)
        for field in dataclasses.fields(record)
        if _states_a_conclusion(field.name) and _is_a_boolean(field.type)
    ]
    return sorted(found, key=lambda item: item.field)


def conclusion_fields_in(
    package: ModuleType, *, settings: Iterable[type] = ()
) -> list[ConclusionField]:
    """Every conclusion boolean on every result record defined under ``package``.

    Reads the records where they are defined rather than where they are
    imported, so a type re-exported from three modules is reported once, under
    the module that owns it.

    ``settings`` names the records that are parameters rather than results, and
    they are skipped. The rule is about what a result says, and a setting saying
    ``exclude_drag`` is asking for a region of a surface to be masked, which is
    the parameter #57 exists to make movable. Refusing it would refuse the
    sweep's own vocabulary.

    Named by the caller rather than guessed at. The two candidate rules are a
    naming convention, which a record can fall out of by being called something
    else, and reading every ``parameters_type`` in the tree, which misses the
    comparison stage because it is not a transform and declares none. So the
    call site says which records are settings, and an addition to that list is a
    line in a diff somebody can disagree with.
    """
    skipped = set(settings)
    found: list[ConclusionField] = []
    for module in _modules_in(package):
        for candidate in vars(module).values():
            if not isinstance(candidate, type) or candidate.__module__ != module.__name__:
                continue
            if not dataclasses.is_dataclass(candidate) or candidate in skipped:
                continue
            found.extend(conclusion_fields(candidate))
    return sorted(found, key=lambda item: (item.record, item.field))


def _modules_in(package: ModuleType) -> list[ModuleType]:
    """``package`` and every module under it, imported.

    A plain module has no ``__path__`` and being handed one is the ordinary
    case rather than a mistake, so the walk starts from an empty path there and
    reads the module itself.
    """
    modules = [package]
    for found in pkgutil.walk_packages(getattr(package, "__path__", ()), f"{package.__name__}."):
        modules.append(importlib.import_module(found.name))
    return modules


def _states_a_conclusion(name: str) -> bool:
    return any(part.startswith(REFUSED_STEMS) for part in name.lower().split("_"))


def _is_a_boolean(annotation: object) -> bool:
    """Whether a field annotation is a boolean.

    The annotation arrives as a string under ``from __future__ import
    annotations``, which every module in this package uses, and as the type
    itself otherwise. Both are read rather than resolved, because resolving one
    means importing whatever the defining module imported, which is a large
    thing to do inside a check.
    """
    if annotation is bool:
        return True
    return isinstance(annotation, str) and annotation.replace(" ", "") in ("bool", "bool|None")
