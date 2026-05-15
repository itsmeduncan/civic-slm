#!/usr/bin/env bash
# Apply civic-slm's branch-protection policy to `main`.
#
# Idempotent — re-running is a no-op when nothing has drifted. Maintainer
# runs this once after merging CI gate changes; GitHub then enforces the
# `ci` checks on every PR. See #55.
#
# Settings applied:
#   - Required status checks: the 4 lint-type-test matrix shards + the
#     `docs link check` job.
#   - strict: branches must be up-to-date with main before merge.
#   - Linear history required (matches project rebase workflow).
#   - No force-push, no branch deletion.
#   - No required PR approvals — solo maintainer; CI is the gate, not a
#     second pair of eyes. (`--admin` bypass remains for unblocking.)
#
# To unset the policy entirely:
#   gh api -X DELETE repos/itsmeduncan/civic-slm/branches/main/protection

set -euo pipefail

OWNER="${OWNER:-itsmeduncan}"
REPO="${REPO:-civic-slm}"
BRANCH="${BRANCH:-main}"

REQUIRED_CONTEXTS=(
  "ubuntu-latest / py3.11"
  "ubuntu-latest / py3.12"
  "macos-14 / py3.11"
  "macos-14 / py3.12"
  "docs link check"
)

# Build the JSON payload via jq so quoting is unambiguous.
payload="$(jq -n \
  --argjson contexts "$(printf '%s\n' "${REQUIRED_CONTEXTS[@]}" | jq -R . | jq -s .)" \
  '{
    required_status_checks: {
      strict: true,
      contexts: $contexts
    },
    enforce_admins: false,
    required_pull_request_reviews: null,
    restrictions: null,
    required_linear_history: true,
    allow_force_pushes: false,
    allow_deletions: false,
    required_conversation_resolution: true,
    lock_branch: false,
    allow_fork_syncing: false
  }')"

echo "Applying branch protection on ${OWNER}/${REPO}@${BRANCH}…"
echo "Required checks: ${REQUIRED_CONTEXTS[*]}"

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${OWNER}/${REPO}/branches/${BRANCH}/protection" \
  --input - <<<"$payload" >/dev/null

echo "Applied. Verifying:"
gh api "repos/${OWNER}/${REPO}/branches/${BRANCH}/protection" \
  --jq '{
    required_status_checks: .required_status_checks.contexts,
    strict: .required_status_checks.strict,
    linear_history: .required_linear_history.enabled,
    force_pushes_blocked: (.allow_force_pushes.enabled | not),
    deletions_blocked: (.allow_deletions.enabled | not)
  }'
