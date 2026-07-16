#!/usr/bin/env bash
# Build WV Sign Mapper as a standalone executable (Mac/Linux).
# Run from the sign_mapper/ directory.
set -e
cd "$(dirname "$0")"

echo
echo "WV Sign Mapper -- Build executable"
echo "===================================="
echo
echo "Output: dist/WVSignMapper/WVSignMapper"
echo "Size:   ~1.5–2.5 GB (PyTorch dominates)"
echo

pip install pyinstaller --quiet
pyinstaller sign_mapper.spec --clean --noconfirm

echo
echo "============================================================"
echo " Build complete!"
echo
echo " Executable folder : dist/WVSignMapper/"
echo " Launch with       : ./dist/WVSignMapper/WVSignMapper"
echo
echo " Zip dist/WVSignMapper/ to distribute to users."
echo "============================================================"
echo
