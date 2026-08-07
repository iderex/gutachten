# compare

Comparison and scoring. The surface is divided into cells, the cells are
correlated, the registration is searched over translation and rotation, and the
congruent matching cells rule decides how many cells agree. This subpackage
turns two preprocessed surfaces into a score and the intermediate quantities
behind it, and it stops there: it does not decide what the score means. That
separation is deliberate, because a comparison module that also draws a
conclusion is one nobody can re-use with a different decision rule.
