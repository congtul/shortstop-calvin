#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT/mdt_policy"

git apply "$REPO_ROOT/patches/mdt_policy_shortstop.patch"

echo "MDT patch applied successfully."
