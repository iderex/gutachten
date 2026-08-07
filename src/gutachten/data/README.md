# data

Acquisition and caching. This is where a named scan is fetched from a public
source into a content addressed cache and looked up again later by that name.
Nothing downloaded is committed, and nothing here runs inside the gate: the code
path that reaches a network lives behind the harness that says so in its name.
Keeping acquisition in one subpackage is what lets the rest of the tree stay
offline by construction rather than by everyone remembering.
