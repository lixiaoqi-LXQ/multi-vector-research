#!/usr/bin/env bash
# Publish the committed Chimera subtree while preserving the original scorer history.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo 'Commit current research changes before synchronizing Chimera.' >&2
  exit 1
fi
anchor=c97f24cc7ea011e0da3fe4a6db808eb0c10590cc
revision=$(git subtree split --prefix=Chimera --onto="$anchor" HEAD)
git push chimera "$revision:refs/heads/Chimera+" baseline:refs/heads/baseline
printf 'Chimera-uma/Chimera+: %s\n' "$revision"
