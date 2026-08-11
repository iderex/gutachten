# The quality target, what was decided about each check in it, and what this repository produces

The merge gate on the public
[jellyfin-plugin-sso](https://github.com/iderex/jellyfin-plugin-sso) repository is
the target for the quality of this one. That set was arrived at by running into
the failures it now refuses, so copying it is cheaper than rediscovering it, and
copying it literally would be wrong: that is a plugin for a media server, written
in a compiled language, with a packaging pipeline and a login flow, and this is a
scientific pipeline in Python with none of those.

So each check in the target gets a decision, and each deviation gets its
reasoning. A check dropped without a reason is a check dropped by accident, and
the difference is invisible a month later.

The decisions are data in [parity.json](parity.json) rather than a list in this
page, and `tests/unit/test_parity.py` reads them. This page is the argument for
their shape and for the last section, which is the one a reader is most likely to
want and the one that rots fastest.

## The target set is derived rather than remembered

    gh api repos/iderex/jellyfin-plugin-sso/rules/branches/main \
      --jq '.[] | select(.type=="required_status_checks")
            | .parameters.required_status_checks[].context'

    build
    ABI floor build
    Package (JPRM) / Build package
    Package (JPRM) / Generate SBOM
    CodeQL
    Analyze (csharp)
    DCO sign-off
    Deterministic PR-hygiene checks
    Enforce greppable invariants
    Reject Trojan Source Unicode
    Audit workflows (zizmor)
    prettier
    dependency-review

Read on 2026-08-11, and unchanged from the reading on 2026-08-07 that
[#105](https://github.com/iderex/gutachten/issues/105) records. It is a live
gate on a repository that is still worked on, so it moves. Re-run the command
rather than quoting this block, and if it has moved, the decisions below are a
record of a target that was, not of the target that is.

## The counts

| what | count |
| --- | --- |
| required checks in the target set | 13 |
| ported unchanged | 4 |
| adapted | 9 |
| check names this repository produces | 13 |
| of those, on a pull request | 11 |

`tests/unit/test_parity.py` reads this table and refuses it when it no longer
matches the file it describes.

Nothing in the target set came out as not applicable. The two things
[#105](https://github.com/iderex/gutachten/issues/105) records as having no
counterpart here, the plugin archive with its repository manifest and the end to
end login check, are not required checks over there, so they are recorded in
[parity.json](parity.json) beside the derived set rather than inside it. Reading
them as part of the thirteen would understate how much of the target this
repository has taken on.

## What the four ported checks have in common, and why they name no issue

A bidirectional control character makes source read as its own opposite in any
language. A workflow with a loose permission or an unpinned action is the same
artefact with the same failure modes here. A newly introduced dependency with a
known vulnerability is the same defect. A commit with no sign-off trailer leaves
the same gap. None of those is about C# or about a media server, so all four
transferred without a change.

They also predate the first issue on this board. All four arrived in the commit
that started the repository:

    for f in dco unicode-guard zizmor dependency-review; do
      printf '%s %s\n' "$(git log --diff-filter=A --format=%h -- .github/workflows/$f.yml)" "$f.yml"
    done

    3270a00 dco.yml
    3270a00 unicode-guard.yml
    3270a00 zizmor.yml
    3270a00 dependency-review.yml

    git log -1 --format=%B 3270a00 | grep -c '#[0-9]'
    0

So their entries in [parity.json](parity.json) name no issue, and that is the
fact rather than an omission. The test requires an issue on everything that is
adapted or has no counterpart on the target board, and it does not require one
here, because there is nothing left to build.

What that costs is worth stating: the four guards landed with no issue arguing
for them, so what a reader has is this page and the comment at the top of each
workflow file, and nothing that says why the four were chosen and not a fifth.

## Where the adapted checks stand

Four of the nine are in force. The build check became a locked install across
three platforms, the formatting check became this language's formatter over the
source, the hygiene check kept its shape and changed its co-change rules, and
the greppable invariants check is the one of the four that is still open. Five
are not in force yet and each has an issue holding it.

Three of those issues did not exist until this record was written, and finding
that is most of what writing it was for. The floor build, the distribution build
and the formatting of the documentation were each named as an adaptation and
each had nothing anywhere to build it, which is exactly the state a parity record
is supposed to make visible rather than the state it is supposed to hide.

Two of the nine sit on issues in earlier milestones rather than in the parity
one, because they landed before that milestone existed. The build matrix is
[#17](https://github.com/iderex/gutachten/issues/17) and the source formatter is
[#19](https://github.com/iderex/gutachten/issues/19), both in Foundations, both
closed as completed. Opening a second issue in the parity milestone for a check
that already runs would produce an issue whose done-condition is already met,
which is not work.

## What this repository has that the target does not

Four checks here protect something that board has nothing like, and they are the
substance of the project rather than an embellishment on it. A recorded run
reproduces to the same scores. A parameter that is not in the manifest cannot
reach the pipeline. No output states a conclusion. No test in the gate touches a
display, a network or an administrator.

Three of the four are in force and reach the merge path through the suite, so
they carry no check name of their own: they are legs of `test (ubuntu)`,
`test (macos)` and `test (windows)`, and the offline claim is measured a second
time by `container`. The fourth, the reproducibility gate, is
[#118](https://github.com/iderex/gutachten/issues/118) and is open.

That is worth being plain about. A property enforced inside the suite is
enforced, and it is also invisible to anybody choosing a required set from a
list of check names, because it has no name on that list. The names it hides
behind are there instead.

Two more checks exist on the target board without being required there, and both
are worth having here for reasons this project has and that one does not: a
suite that would not notice a changed number is not a check on a number, and the
parser of an untrusted container is where this project is attackable. They are
[#112](https://github.com/iderex/gutachten/issues/112) and
[#115](https://github.com/iderex/gutachten/issues/115).

## The check names this repository produces

This is the list the decision about which checks hold a merge is made from, and
it is the reason a check name is fixed in its workflow file rather than left to
default. A renamed check drops silently out of a required set and the merge it
was protecting goes through green.

The names are in [parity.json](parity.json) and the test derives them from the
workflow files rather than trusting them, in both directions: a name in the
record that no workflow produces is refused, and a name a workflow produces that
the record does not carry is refused. That second direction is the one that
matters, because a set chosen from a list with an entry missing is a set with a
hole in it.

Eleven names appear on a pull request. Two do not: `Scorecard analysis` runs on a
push to the default branch and on a schedule, and `fuzz session` runs on a
schedule and on request. Only a check that runs on a pull request can hold a
merge, so those two are on the list and are not candidates for the required set.

One name is not derived from anything in this tree. The code scanning run named
`zizmor` is created by the SARIF upload rather than by a job, so no reading of
the workflow files produces it, and the test asserts that the record says where
it comes from instead of asserting that it exists. It is also conditional: the
upload step is skipped where the token cannot write security events, which is a
fork pull request and a Dependabot one, so on those the check run is absent while
the workflow audit still refuses findings. A required set containing that name
would block every fork pull request, and this sentence is the only warning of
that a reader gets.

The list read from the workflows, and the same list as it appeared on a real
pull request:

    gh api repos/iderex/gutachten/commits/f0dea69/check-runs \
      --jq '.check_runs[] | "\(.name)\t\(.app.slug)"' | sort -u

    Audit workflows (zizmor)	github-actions
    container	github-actions
    DCO sign-off	github-actions
    dependency-review	github-actions
    lint	github-actions
    pull request hygiene	github-actions
    Reject Trojan Source Unicode	github-actions
    test (macos)	github-actions
    test (ubuntu)	github-actions
    test (windows)	github-actions
    zizmor	github-advanced-security

Read on 2026-08-11 against the head commit of pull request #179. Ten job names
and one code scanning run.

## What is not decided here

Which of these names holds a merge, and whether a review is required, is set on
the repository rather than in a pull request. The gate today refuses branch
deletion and force pushes and requires a pull request with zero approving
reviews, and it requires no status check at all:

    gh api repos/iderex/gutachten/rulesets/20530438 \
      --jq '{enforcement, bypass: .bypass_actors, rules: [.rules[].type]}'
    {"bypass":[],"enforcement":"active","rules":["deletion","non_fast_forward","pull_request"]}

Read on 2026-08-11. So a red build does not hold a merge on this repository
today, and every check above is advisory until somebody with settings access
says otherwise. That is
[#2](https://github.com/iderex/gutachten/issues/2), under which checks hold a
merge, and this page is the list that question was waiting for.
