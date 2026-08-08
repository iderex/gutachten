# The image the gate runs in when it has to prove the conditions rather than
# assert them: no display server, an unprivileged user, and no route out.
#
# The dependencies are installed here, while the network is still available,
# because `uv sync` reaches an index and the run that follows must not be able
# to. Splitting the two is the whole point of the file. What the suite is then
# asked to do is run against an environment where reaching a host is impossible,
# rather than against one where it merely chose not to.
#
# Both images are pinned by digest and carry their tag in a comment, which is
# the posture every action in this repository's workflows already has. A tag is
# a moving reference, and a base image that moved under a green run makes that
# run unreproducible in the one job whose subject is reproducibility of the
# environment.

FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

# The same uv version the other jobs install, so this one does not resolve
# through a different resolver from the rest of the gate.
COPY --from=ghcr.io/astral-sh/uv:0.12.2@sha256:069a51314a7bb6031777a9273205fe1b0b19e914ef418207d1338b268df641dd /uv /usr/local/bin/uv

# The interpreter is the image's own. Without this uv would fetch one of its
# own choosing, which would make the version under test a property of what uv
# felt like downloading rather than of the range pyproject.toml declares.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/src/.venv \
    PATH=/src/.venv/bin:$PATH

WORKDIR /src
COPY . /src

# `--locked` here for the reason the other jobs use it: a green run against an
# unlocked resolution says nothing about what anybody else will install.
RUN uv sync --locked

# The run is unprivileged and pytest and coverage both write beside the tree, so
# the tree has to be writable by a user that is not the one that built it. Doing
# this rather than running as root is the point: a suite that only passes as
# root has not been shown to pass unelevated.
RUN chmod -R a+rwX /src

CMD ["python", "-m", "pytest"]
