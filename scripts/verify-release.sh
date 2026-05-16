#!/usr/bin/env bash
# Verify a civic-slm release artifact's Sigstore signature.
#
# Closes #27. Two checks:
#   1. Every file's SHA256 matches the manifest (`shasum -a 256 -c`).
#   2. The manifest's cosign bundle was issued to the canonical maintainer
#      identity by the canonical OIDC provider.
#
# A green run means: the bytes on disk are what the maintainer signed and
# the signer was the holder of the pinned identity at sign time. Both
# checks must pass or the script exits non-zero.
#
# Usage:
#   scripts/verify-release.sh <artifact-dir>
#
# Override the expected identity for forks / personal releases:
#   CIVIC_SLM_RELEASE_IDENTITY=alice@example.org \
#   CIVIC_SLM_RELEASE_OIDC_ISSUER=https://accounts.google.com \
#       scripts/verify-release.sh ./downloaded-model

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <artifact-dir>" >&2
  exit 2
fi

art_dir="$1"
manifest="$art_dir/sha256sums.txt"
bundle="$art_dir/sha256sums.txt.bundle"

if [[ ! -d "$art_dir" ]]; then
  echo "error: $art_dir is not a directory" >&2
  exit 1
fi
if [[ ! -f "$manifest" ]]; then
  echo "error: missing $manifest — was this artifact signed?" >&2
  exit 1
fi
if [[ ! -f "$bundle" ]]; then
  echo "error: missing $bundle — was this artifact signed?" >&2
  exit 1
fi

if ! command -v cosign >/dev/null 2>&1; then
  echo "error: cosign not installed. macOS: brew install cosign" >&2
  echo "       linux: see https://docs.sigstore.dev/system_config/installation/" >&2
  exit 127
fi

# Pinned values — these are the canonical maintainer release identity.
# Forks/redistributors override via env. Keep in sync with RELEASING.md
# and scripts/sign-release.sh.
IDENTITY="${CIVIC_SLM_RELEASE_IDENTITY:-itsmeduncan@gmail.com}"
OIDC_ISSUER="${CIVIC_SLM_RELEASE_OIDC_ISSUER:-https://github.com/login/oauth}"

echo ">> $art_dir"

echo "   [1/2] re-hashing files…"
(
  cd "$art_dir"
  # `shasum -c` reads lines like `<hex>  <path>` and re-hashes <path>.
  # --quiet prints only failures.
  shasum -a 256 --check --quiet sha256sums.txt
)
echo "         ✓ all files match the manifest"

echo "   [2/2] cosign verify-blob…"
cosign verify-blob \
  --bundle "$bundle" \
  --certificate-identity "$IDENTITY" \
  --certificate-oidc-issuer "$OIDC_ISSUER" \
  "$manifest" >/dev/null
echo "         ✓ manifest signed by $IDENTITY via $OIDC_ISSUER"

echo
echo "VERIFIED."
