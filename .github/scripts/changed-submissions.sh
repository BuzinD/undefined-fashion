#!/usr/bin/env bash
# Print the submission folders to score, one name per line.
#
#   changed-submissions.sh all          every submission
#   changed-submissions.sh A...B        submissions changed in that git range
set -euo pipefail

if [ "$1" = "all" ]; then
  names=$(ls submissions)
else
  names=$(git diff --name-only "$1" -- submissions/ | cut -d/ -f2 | sort -u)
fi

for name in $names; do
  # Keep plain folder names that still have a model.py (a deleted folder is skipped).
  if [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] && [ -f "submissions/$name/model.py" ]; then
    echo "$name"
  fi
done
