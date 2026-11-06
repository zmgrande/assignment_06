#!/bin/sh
# graderthan sandbox entrypoint: install this assignment's dependencies, run the
# autograder, and emit /work/.grader/results.json
#
# NOTE: the pip install below needs "Sandbox network" turned ON for the
# assignment in GraderThan. With network off it fails and the tests will error
# on any import that is not already in the base image.
cd /work || exit 1

# --break-system-packages is required on PEP 668 ("externally managed") images
# and is rejected by pip < 23.0, so fall back to a plain install. A failure here
# is deliberately NOT fatal: grade.py still runs, so a missing dependency shows
# up as a normal test failure with a readable message instead of a dead run.
if [ -f requirements.txt ]; then
    python3 -m pip install -q --break-system-packages -r requirements.txt 2>/dev/null \
        || python3 -m pip install -q -r requirements.txt 2>/dev/null \
        || echo "WARNING: dependency install failed (is Sandbox network on?)" >&2
fi

# This assignment calls the CENT iSchool IoT Portal for real. The grader's key lives
# only in the private solution repo (grader/.cent_api_key, never in the student
# repo); when it is present, hand it to the tests as CENT_API_KEY.
if [ -f grader/.cent_api_key ]; then
    CENT_API_KEY="$(tr -d '[:space:]' < grader/.cent_api_key)"
    export CENT_API_KEY
fi

exec python3 grader/grade.py
