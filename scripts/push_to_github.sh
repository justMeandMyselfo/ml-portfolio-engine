#!/usr/bin/env bash
# Create a GitHub repo and push this project to it.
#
# Usage:
#   GITHUB_TOKEN=ghp_xxx GITHUB_USER=yourname ./scripts/push_to_github.sh [repo-name] [public|private]
#
# Requires a GitHub Personal Access Token with "repo" scope (classic) or
# "Contents: read & write" + "Administration: read & write" (fine-grained).

set -euo pipefail

REPO_NAME="${1:-ml-portfolio-engine}"
VISIBILITY="${2:-public}"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a Personal Access Token}"
: "${GITHUB_USER:?Set GITHUB_USER to your GitHub username}"

PRIVATE=false
[ "$VISIBILITY" = "private" ] && PRIVATE=true

echo "Creating repo $GITHUB_USER/$REPO_NAME (private=$PRIVATE) ..."
curl -sS -X POST https://api.github.com/user/repos \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"name\":\"$REPO_NAME\",\"private\":$PRIVATE,\"description\":\"ML-enhanced portfolio optimization engine: Markowitz vs HMM regimes + Random Forest Black-Litterman\"}" \
  >/dev/null || echo "(repo may already exist; continuing)"

cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  git init -b main
fi
git add .
git -c user.email="${GIT_EMAIL:-you@example.com}" -c user.name="${GIT_NAME:-$GITHUB_USER}" \
  commit -m "Initial commit: ML-enhanced portfolio optimization engine" || echo "(nothing to commit)"

REMOTE="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push -u origin main

echo "Done: https://github.com/${GITHUB_USER}/${REPO_NAME}"
