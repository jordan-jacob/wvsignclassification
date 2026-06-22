#!/usr/bin/env bash
# Manually push all checkpoints to GitHub via Git LFS.
set -e
cd "$(dirname "$0")/.."

git lfs install
git add checkpoints/*.pt
git commit -m "Add trained checkpoints"
git push origin main
