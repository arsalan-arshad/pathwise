#!/usr/bin/env bash
# Runs smoke.mts against the real `ai` and `@langchain/core` packages, installed in a throwaway temp directory.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
npm init -y >/dev/null
npm install --no-audit --no-fund ai @langchain/core >/dev/null
cp "$here/smoke.mts" smoke.mts
PW_SRC="$repo/src" "$repo/node_modules/.bin/tsx" smoke.mts
