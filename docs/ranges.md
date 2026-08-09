# The plausible range of every parameter, and where each bound came from

A sensitivity result is worth exactly as much as the range it was taken over. A
range narrow enough to keep the answer stable is a way of not measuring, and a
range wide enough to include settings no analyst would use produces an alarming
number any practitioner can dismiss. Neither failure is visible in the final
figure. So the ranges are an artefact in their own right, they are declared
before the sweep that reads them exists, and every bound says where it came from.

The declaration is [ranges.json](ranges.json). This page is the argument for its
shape, the counts, and what the weakest part of it is.

## Where the file is, and why it is not in profiles/

`gutachten.profile.load_directory` globs `profiles/*.json` and reads every file
it finds as a profile, so a file there that is not one makes the loader refuse
the directory. The declaration sits beside this page instead. It is data rather
than prose for the same reason a profile is: the check below reads it, and the
sweep will read it rather than reading a table out of a document.

## What a range is here

Each parameter carries one of two shapes.

An **interval** has a lower bound and an upper bound. Each bound carries a
`source` out of a closed vocabulary and a `where` sentence naming what it came
from.

A **set** lists the admissible values, each with the same two fields. Six of the
parameters take a word and two take a flag, and there is nothing for a bound to
be. Writing them as an interval of zero to one would read as a declared range and
sweep nothing, and leaving them out because they have no bound is the likelier
accident: whether the form model is a plane or a second order polynomial is the
parameter the study most needs to move, and `exclude_drag` is the one issue #57
exists to make movable at all.

Twelve parameters admit null as well. A null is declared with its own source and
sentence, because a nullable parameter whose null is undeclared has a state the
design cannot reach. In every case here the null is not free: the step refuses it
stated beside the wrong companion, so it is fixed by another parameter rather
than chosen. The sentences say which.

Where the literature gives a single value and no range, that is recorded in a
`literature` field. It is a finding about the literature rather than a bound, and
it tells a reader that the bracket around that value is ours.

## The four words a source can be

- `published`, a value stated in a published method
- `standardised`, a range stated in a standard
- `physical`, the limit of the measurement itself
- `judgement`, somebody here decided it

The vocabulary is closed and the check below refuses a fifth word, because a
fifth word is a bound escaping the count that says how much of this file is our
own opinion.

Two boundaries inside it are worth stating, because they decide the counts.

A limit the code refuses is counted as a `judgement`, not as a `physical` one.
`gutachten.transforms.level` refuses a polynomial of order one because a first
order polynomial is a plane under a second name. That is a choice made in this
repository, and dressing it as a limit of the measurement would inflate the class
that is meant to be the strongest.

A value carried across from a published method under a different name is counted
as `published`, and the `where` sentence says what the mapping was. The reference
chain crops the `exterior`; this tree's edge trim calls the same operation
`frame`. Counting that as a judgement would hide the one thing about it a reader
can check.

## The counts

    .venv/Scripts/python.exe -c "
    import json, collections, pathlib
    d = json.loads(pathlib.Path('docs/ranges.json').read_text(encoding='utf-8'))['parameters']
    def bounds(e):
        out = [e['lower'], e['upper']] if e['kind'] == 'interval' else list(e['values'])
        return out + ([e['null']] if 'null' in e else [])
    c = collections.Counter(b['source'] for e in d.values() for b in bounds(e))
    print('parameters:', len(d))
    print('bound entries:', sum(len(bounds(e)) for e in d.values()))
    print('by source:', dict(sorted(c.items())))
    print('with a literature note:', sum(1 for e in d.values() if 'literature' in e))
    "
    parameters: 40
    bound entries: 93
    by source: {'judgement': 61, 'physical': 26, 'published': 6}
    with a literature note: 9

Run against `docs/ranges.json` at the commit this page landed on. `uv` is not on
the path of the machine this was read on, so the project virtualenv was called
directly; it is the same interpreter and the same installed environment the gated
command runs.

The counted unit is a bound entry rather than a parameter: an interval
contributes two, a set contributes one per admissible value, and a nullable
parameter contributes one more. Forty parameters therefore produce ninety three
entries.

| source | entries |
| --- | --- |
| `published` | 6 |
| `standardised` | 0 |
| `physical` | 26 |
| `judgement` | 61 |

`tests/unit/test_ranges.py` reads this table and refuses it when it no longer
matches the file it describes. A count written once and left behind is the
failure this repository calls restated-not-referenced, and it lands hardest on
the last row.

## What those counts say, plainly

Sixty one of ninety three bounds are this repository's own judgement. Twenty six
come from the measurement, and most of those are the sampling interval, the
extent of the field, and the fact that a proportion cannot exceed one. Six come
from a published method. None comes from a standard.

The last of those is not an oversight. The filter this project bandpasses with is
ISO 16610-71:2014, the text of that standard is not public, and
[filtering.md](filtering.md) records that its internally fixed constants could not
be read. So the robust tuning constant, which is the one parameter here a standard
plausibly does fix, carries a judgement bound and says so.

What that means for a reader of the sensitivity report is worth stating in the
same place as the numbers rather than in a footnote. The ranges are the axis every
index in that report is computed along, and two thirds of them were chosen here.
A result saying a parameter is unimportant is a result about the interval it was
moved over, and that interval is mostly ours.

## What the check does and does not do

`tests/unit/test_ranges.py` refuses:

- a parameter a sweep can reach with no declared range
- a declaration naming a parameter nothing answers to
- a bound with a source outside the four words, or with no sentence
- a set declared for a number, or an interval declared for a word or a flag
- a nullable parameter with no null declared, or a null declared for a parameter
  that refuses one
- a flag whose set holds one value, which is a setting held fixed while the
  report says it moved
- an interval running downwards, or fractional bounds on a whole number parameter
- counts in this page that disagree with the file

What is reachable is derived rather than listed. The registry is the authority
for the preprocessing parameters. The comparison parameters are found by
inspecting `gutachten.compare.register` and `gutachten.compare.cmc` for their
parameter records, because those are not transforms and register nowhere, and a
check reading only the registry would pass a declaration that covered the
preprocessing and none of the scoring.

It does not check that a bound is the right one. That is a judgement, and the
`source` field is there so a reader can see which bounds nobody outside this
repository has ever stated.

It does not check that a value in a set is one the step admits, where the field
is typed as a plain string. Three of the seven sets are: `level.model`,
`trim-edge.criterion` and the two masking methods name their values as strings
because a parameter record is written into the manifest as plain data. The
enumerations behind them are in the transform modules, and matching a field to
one of them by convention would be a check that guessed. Where the field is typed
as an enumeration or as a flag, the admissible set is derived and is checked.

## What is not decided here

Which of these parameters a given design moves, how the joint space is sampled
and how large the sample is, are the design and are
[#79](https://github.com/iderex/gutachten/issues/79). This page fixes the axes it
will be drawn over and nothing about how it is drawn.
