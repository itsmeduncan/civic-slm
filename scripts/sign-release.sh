#!/usr/bin/env bash
# Sign a civic-slm release artifact with Sigstore (keyless OIDC).
#
# Closes #27. The signed thing is a `sha256sums.txt` manifest — one
# signature covers every file in the artifact directory. Users verify
# the manifest signature, then verify each file's sha256 against the
# manifest. Pattern matches Debian / Sigstore upstream releases.
#
# What signs: cosign sign-blob in keyless mode. An interactive browser
# OAuth flow against the configured OIDC provider mints a short-lived
# Fulcio certificate bound to the maintainer's identity. No private key
# is kept on disk; nothing in this repo to rotate.
#
# Outputs (in the artifact dir):
#   - sha256sums.txt        — sorted SHA256 manifest, one file per line
#   - sha256sums.txt.bundle — cosign bundle (sig + cert + Rekor entry)
#
# Usage:
#   scripts/sign-release.sh artifacts/civic-slm-v1-mlx-q4
#   scripts/sign-release.sh artifacts/civic-slm-v1-mlx-q4 artifacts/civic-slm-v1-gguf-q5km
#
# Env overrides (defaults match the canonical maintainer release identity
# documented in RELEASING.md — change here AND in scripts/verify-release.sh
# AND in RELEASING.md if rotating):
#   CIVIC_SLM_RELEASE_IDENTITY     default itsmeduncan@gmail.com
#   CIVIC_SLM_RELEASE_OIDC_ISSUER  default https://github.com/login/oauth
#
# Re-running over an already-signed directory replaces the existing
# manifest and bundle (idempotent — useful when a file changes and you
# need to re-sign).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <artifact-dir> [<artifact-dir>...]" >&2
  echo "       sign one or more release artifact directories." >&2
  exit 2
fi

if ! command -v cosign >/dev/null 2>&1; then
  echo "error: cosign not installed. macOS: brew install cosign" >&2
  echo "       linux: see https://docs.sigstore.dev/system_config/installation/" >&2
  exit 127
fi

OIDC_ISSUER="${CIVIC_SLM_RELEASE_OIDC_ISSUER:-https://github.com/login/oauth}"
# Identity is captured by the cert during sign; the verify script is what
# pins it. We don't enforce it at sign time — the signer's interactive
# OAuth picks the identity automatically.

for art_dir in "$@"; do
  if [[ ! -d "$art_dir" ]]; then
    echo "error: $art_dir is not a directory" >&2
    exit 1
  fi

  manifest="$art_dir/sha256sums.txt"
  bundle="$art_dir/sha256sums.txt.bundle"

  echo ">> $art_dir"

  # Generate the manifest. Sort by path so the file is byte-deterministic
  # — the same artifact directory always hashes to the same manifest,
  # regardless of filesystem iteration order. Paths are relative to the
  # artifact dir (not the repo) so a downloaded tarball verifies cleanly.
  echo "   computing sha256sums.txt"
  ( cd "$art_dir" && \
    find . -type f \
      ! -name 'sha256sums.txt' \
      ! -name 'sha256sums.txt.bundle' \
      -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 \
    | sed 's|  \./|  |' \
    > sha256sums.txt
  )

  n_files=$(wc -l < "$manifest" | tr -d ' ')
  echo "   $n_files files in manifest"

  echo "   cosign sign-blob (keyless OIDC: $OIDC_ISSUER)"
  cosign sign-blob \
    --yes \
    --oidc-issuer "$OIDC_ISSUER" \
    --bundle "$bundle" \
    "$manifest" >/dev/null

  echo "   signed: $bundle"
done

echo
echo "Done. Next steps:"
echo "  1. Upload sha256sums.txt + sha256sums.txt.bundle alongside the"
echo "     model files when pushing to HF Hub."
echo "  2. Attach both files to the GitHub Release for the tag."
echo "  3. Verify: scripts/verify-release.sh <artifact-dir>"
