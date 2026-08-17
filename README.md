# gutachten

Firearm toolmark identification is still largely visual, and a 2009 National Academies report questioned its objectivity and asked for objective criteria and error rate estimates. The infrastructure answered: the NIST Ballistics Toolmark Research Database is open, the Open Forensic Metrology Consortium exists, and X3P to ISO 25178-72 is the exchange format with open readers. The analysis side did not. The CMC algorithm classified all 433 matching and 4812 non-matching pairs correctly in one study, but a long manually configured preprocessing hangs on that number with free parameters whose influence on the score nobody has mapped. An FBI blind study over 8640 comparisons found false positive rates under one per cent, but most errors came from a few examiners and the per-person probabilities differ, while courts hear the pooled rate. This board builds the open parameterised pipeline on public data, the sensitivity analysis, and a likelihood ratio with an uncertainty.

Planning happens on the issue tracker first. Every decision that shapes
the architecture is written down there with its reasons before the code
that depends on it exists.

## Running it

Nothing is released yet and there is no operator command. What exists is a tree
you can install and test:

```
uv sync --locked
uv run pytest
```

[CONTRIBUTING.md](CONTRIBUTING.md) is the full version of that, starting from a
machine with nothing installed, and it also says what to run before pushing.

This repository has no license file. Until one is chosen, default copyright
applies, which means it cannot lawfully be used, forked or modified. That is a
question for the maintainer and it is open.

See [SECURITY.md](SECURITY.md) for what to report privately and what to report in
the open, and [NOTICE.md](NOTICE.md) for the intended-use notice.

## License

AGPL-3.0, copyright 2026 Nils Lehnen.

The full text is in [LICENSE](LICENSE).
