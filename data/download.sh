#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# download.sh – fetch the LUNA16‑DP2D dataset
# Usage:   cd data && bash download.sh
# ──────────────────────────────────────────────────────────

set -euo pipefail

REPO_ID="nourbourai/luna16-dp2d"
TARGET_DIR="."   # downloads into the current directory (data/)

echo "📥 Downloading dataset: $REPO_ID"
echo "   Target: $(realpath "$TARGET_DIR")"

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download "$REPO_ID" \
        --repo-type dataset \
        --local-dir "$TARGET_DIR" \
        --local-dir-use-symlinks False
    echo "✅ Download complete."
else
    echo "⚠️  huggingface-cli not found."
    echo ""
    echo "Install it with:"
    echo "    pip install huggingface_hub"
    echo ""
    echo "Then re-run this script, or manually download from:"
    echo "    https://huggingface.co/datasets/$REPO_ID"
    echo ""
    echo "Alternatively, use Zenodo (DOI: 10.5281/zenodo.20314843):"
    echo "    Download all .zip parts and run:"
    echo "        cat luna16-dp2d.zip.* > luna16-dp2d.zip"
    echo "        unzip luna16-dp2d.zip -d ."
    exit 1
fi
