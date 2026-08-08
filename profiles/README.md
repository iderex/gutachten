# profiles

Named parameter sets for the pipeline, one file per profile.

A profile fixes every preprocessing parameter so that a run can be named rather
than described, and so that two people comparing results can establish in one
word whether they ran the same thing. One profile reproduces the configuration
of the published work this project is measuring against, which is what makes the
reproduction check a comparison rather than an approximation.

A parameter that is not in a profile is a parameter somebody chose silently, and
that is the failure this directory exists against.

## What a file holds

A name, a version, one sentence of description, and the chain as an ordered list
of steps. Each step names a registered transform, sets every parameter that
transform declares, and records where each of those values came from. The reader
is `gutachten.profile`, which refuses a file that could not run as written: a
parameter unset, a parameter the step does not take, a value of the wrong type, a
value with no recorded provenance, or a chain whose steps cannot run in the order
given.

The file name and the recorded name have to agree, because a profile is selected
by one and found by the other.

Every step this tree registers is named by at least one profile here, and the
suite refuses a registered step that none of them runs. That is what makes a
parameter added to a step red the build: the profiles resolve every step, so a
new field is a field some file in this directory does not set, and the reader
above refuses it with the step, the field and the profile named. A step no
profile runs would escape that, because nothing would ever resolve its record
against a file. `every-step` is the chain that exists to keep the set covered.

## Where a value came from

Every parameter carries three fields. `origin` is one of three words and nothing
else:

- `stated`, the named source gives this value for this parameter
- `adapted`, the source gives something this was converted from, and `where` says
  what the conversion was
- `not-sourced`, nothing states it, and `where` says what was used instead

`where` names the source or the substitute, and `confidence` says what the entry
is worth. Free prose alone would collapse the three into something nobody can
count, and counting is the point: how much of a published chain is actually
published is a result about the literature, not a footnote.

## What the two profiles are

`published-chain` carries the preprocessing of the open reference implementation
of the congruent matching cells method across to the steps this tree registers.
It is not the paper's own statement of its values. What could be read was that
implementation's documented chain and defaults, and the sources in the file name
the pages and the date they were read.

Three things it does not reproduce, which matter more than the values it does
carry.

The reference crops the interior of the scan to remove the firing pin
impression. No such step is registered here yet, so the chain omits it rather
than approximating it; that step is issue #58.

The reference measures its crops in samples and this project's edge trim takes a
length. Converting one to the other needs the sampling interval of the scan,
which a profile cannot know, so the width in this profile is 30 samples only on a
surface sampled at 4 micrometres and is a different amount of surface on any
other scan.

The reference downsamples before comparing. That is not a step in this tree at
all.

`every-step` is a second configuration that runs every step this tree registers.
It exists so that a chain longer than the reproduction chain is a profile rather
than a code change, and none of its values is anybody's published one. Neither
file is the correct configuration. A repository that ships one blessed profile
teaches the habit this project is arguing against.

## What is not decided here

How an operator selects a profile is the operator surface, issue #122. Whether a
built distribution carries this directory is issue #132.
