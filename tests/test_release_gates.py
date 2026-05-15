"""Release-readiness gates — mechanical enforcement of project invariants.

These are the invariants the MODEL_CARD and audit-gate rely on. Today they
live in maintainer vigilance; once we tag v1 and others reproduce the
numbers, drift becomes "can't reproduce" instead of "the maintainer
remembered." This file moves them from tribal knowledge to `git rejects the
PR`. See #55.

The tests intentionally don't deduplicate with `_load_recipes()` — we want
to fail at the YAML level, not at "recipe registry threw."
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "src" / "civic_slm" / "ingest" / "recipes"
SOURCES_MD = REPO_ROOT / "docs" / "SOURCES.md"

# `## <slug> (<STATE>)` — slug is kebab-case lowercase, state is two-letter caps.
# The literal `## <jurisdiction-slug> (<STATE>)` inside the doc's audit-template
# code fence on line 13 doesn't match (angle-bracket placeholders).
_SOURCES_HEADING_RE = re.compile(r"^## ([a-z][a-z0-9-]*) \(([A-Z]{2})\)\s*$", re.MULTILINE)

# Match the *first* Decision line per section, capture the verdict word.
# Variants seen in the file:
#   `- **Decision:** **GO.** Maintainer override: ...`
#   `- **Decision:** **GO** under the v0.2.x ...`
#   `- **Decision:** **PENDING.** Recipe ...`
_DECISION_RE = re.compile(
    r"^- \*\*Decision:\*\*\s+\*{0,2}(GO|NO-GO|PENDING)\b",
    re.MULTILINE,
)
_VALID_DECISIONS = {"GO", "NO-GO", "PENDING"}


def _yaml_recipe_slugs() -> set[str]:
    """Slugs from `src/civic_slm/ingest/recipes/*.yaml`. File stem, not the
    `jurisdiction:` field — those should agree but the test for that is
    a separate concern handled in `tests/test_recipes.py`.
    """
    return {p.stem for p in RECIPES_DIR.glob("*.yaml")}


def _python_recipe_slugs() -> set[str]:
    """Slugs from `*.py` recipes, excluding helpers/templates that aren't real."""
    out: set[str] = set()
    for p in RECIPES_DIR.glob("*.py"):
        if p.stem.startswith("_") or p.stem == "yaml_recipe":
            continue
        # File names like `san_clemente.py` correspond to slug `san-clemente`.
        out.add(p.stem.replace("_", "-"))
    return out


def _sources_slugs() -> set[str]:
    text = SOURCES_MD.read_text(encoding="utf-8")
    return {m.group(1) for m in _SOURCES_HEADING_RE.finditer(text)}


# ---------------------------------------------------------------------------
# Recipe / audit consistency
# ---------------------------------------------------------------------------


def test_every_yaml_recipe_parses() -> None:
    """A typo in a shipped YAML is caught here, not at first-crawl-time.

    Walks every `*.yaml` in the recipes dir, runs `YamlRecipe.from_yaml`,
    expects no exceptions and that all `{placeholder}` slots except the
    per-call `{since}` / `{max_docs}` are resolved.
    """
    from civic_slm.ingest.recipes.yaml_recipe import YamlRecipe

    yamls = sorted(RECIPES_DIR.glob("*.yaml"))
    assert yamls, "no YAML recipes found — RECIPES_DIR wrong?"

    for path in yamls:
        r = YamlRecipe.from_yaml(path)
        assert r.jurisdiction == path.stem, f"{path}: jurisdiction mismatches filename"
        assert len(r.state) == 2 and r.state.isupper(), f"{path}: state must be 2 caps"
        assert r.start_url.startswith(("http://", "https://")), f"{path}: start_url not http"
        # `{start_url}` must already be resolved by from_yaml.
        assert "{start_url}" not in r.instruction, f"{path}: unresolved {{start_url}}"


def test_every_recipe_has_sources_entry() -> None:
    """Every shipped recipe must appear in `docs/SOURCES.md`, and every
    SOURCES.md heading must correspond to a real recipe. Catches leftover
    audit rows after a recipe is renamed or removed.
    """
    recipe_slugs = _yaml_recipe_slugs() | _python_recipe_slugs()
    sources_slugs = _sources_slugs()

    missing_audit = recipe_slugs - sources_slugs
    orphan_audit = sources_slugs - recipe_slugs

    msg: list[str] = []
    if missing_audit:
        msg.append(
            f"recipes without a SOURCES.md entry: {sorted(missing_audit)}. "
            f"Add a `## <slug> (<STATE>)` section to docs/SOURCES.md."
        )
    if orphan_audit:
        msg.append(
            f"SOURCES.md entries without a matching recipe: {sorted(orphan_audit)}. "
            f"Either restore the recipe or remove the audit row."
        )
    assert not msg, "\n".join(msg)


def test_sources_decision_states_are_known() -> None:
    """Each per-jurisdiction section's first `Decision:` line must resolve to
    one of GO / NO-GO / PENDING. Catches typos like `**Decision:** Go.` that
    wouldn't trip a grep.
    """
    text = SOURCES_MD.read_text(encoding="utf-8")

    # Find heading positions and slice between them to scope each Decision
    # to its section. Anything outside a per-jurisdiction section (e.g. the
    # template/preamble) is ignored.
    headings = list(_SOURCES_HEADING_RE.finditer(text))
    spans = [
        (m.group(1), m.start(), headings[i + 1].start() if i + 1 < len(headings) else len(text))
        for i, m in enumerate(headings)
    ]

    unresolved: list[str] = []
    for slug, start, end in spans:
        section = text[start:end]
        m = _DECISION_RE.search(section)
        if not m:
            unresolved.append(f"{slug}: no recognizable `- **Decision:** <state>` line")
            continue
        if m.group(1) not in _VALID_DECISIONS:
            unresolved.append(f"{slug}: decision={m.group(1)!r} (expected GO|NO-GO|PENDING)")

    assert not unresolved, "\n".join(unresolved)


# ---------------------------------------------------------------------------
# Raw-binary / accidental-artifact gate
# ---------------------------------------------------------------------------

_FORBIDDEN_TRACKED_SUFFIXES = (
    ".bin",
    ".pdf",
    ".safetensors",
    ".gguf",
    ".mp3",
    ".mp4",
    ".wav",
    ".m4a",
    ".webm",
)


def test_no_raw_binaries_committed() -> None:
    """`data/raw/` is gitignored, but a `git add -f` could slip a PDF in.
    Belt-and-braces beyond `.gitignore`: explicitly fail if any tracked
    path under `data/raw/` has a binary suffix.

    `manifest.jsonl` and `.gitkeep` are the only allowed tracked files.
    """
    proc = subprocess.run(
        ["git", "ls-files", "data/raw/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in proc.stdout.splitlines() if line.strip()]
    allowed = {"data/raw/manifest.jsonl", "data/raw/.gitkeep"}
    offending = [
        p for p in tracked if p not in allowed and p.lower().endswith(_FORBIDDEN_TRACKED_SUFFIXES)
    ]
    assert not offending, (
        f"binary files committed under data/raw/: {offending}. "
        f"data/raw/ is gitignored for a reason; use `git rm` to untrack."
    )


# ---------------------------------------------------------------------------
# PII gate over committed eval + SFT
# ---------------------------------------------------------------------------

# PII patterns are scoped tight on purpose. A civic corpus is *full* of
# addresses, .gov phone numbers, and staff emails — those are public records
# and they are exactly what the extraction bench is supposed to extract.
# We only flag patterns that have no business being in a public-records
# corpus and would force a re-release if uploaded to HF Hub.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # SSN — there is no civic public-record reason to ship one.
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit-card-shape (4 groups of 4 digits, space/dash separated). Loose
    # on purpose; collisions are rare enough that a hit deserves human eyes.
    ("credit-card-shape", re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b")),
    # DOB markers — staff reports don't put dates of birth on the public
    # record; finding one suggests an upstream scrub failed.
    ("DOB marker", re.compile(r"\b(?:DOB|D\.O\.B\.|Date of Birth)\s*[:=]", re.IGNORECASE)),
]

# Personal-email-domain check: a `.gov` / `.edu` / `.org` address in a staff
# report is a public-record contact; a gmail/yahoo/hotmail/outlook in the
# committed corpus suggests we scraped a private commenter's email by mistake.
_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
_PERSONAL_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "aol.com",
    "protonmail.com",
}


def _scan_jsonl_for_pii(path: Path) -> list[tuple[int, str, str]]:
    """Return (line_no, kind, sample) tuples for every PII match in `path`.

    Cheap: regex over the raw text of each line. We don't try to parse
    JSON because the goal is "no PII anywhere in this corpus," not "PII
    only in known text fields."
    """
    hits: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            for kind, pat in _PII_PATTERNS:
                m = pat.search(line)
                if m:
                    hits.append((i, kind, m.group(0)))
            for m in _EMAIL_RE.finditer(line):
                domain = m.group(1).lower()
                if domain in _PERSONAL_EMAIL_DOMAINS:
                    hits.append((i, "personal email", m.group(0)))
    return hits


def _committed_jsonl(dirs: list[Path]) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", *(str(d.relative_to(REPO_ROOT)) for d in dirs)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in proc.stdout.splitlines() if line.endswith(".jsonl")]


def test_no_pii_in_committed_eval_and_sft() -> None:
    """Eval examples must be synthetic or scrubbed; SFT pairs likewise. A
    leak here would force a re-release after HF Hub upload.

    Failure mode is loud: prints the first few offending matches with
    line numbers so the contributor can grep and scrub.
    """
    targets = _committed_jsonl([REPO_ROOT / "data" / "eval", REPO_ROOT / "data" / "sft"])
    assert targets, "no committed eval/SFT jsonl files — paths wrong?"

    all_hits: list[str] = []
    for path in targets:
        hits = _scan_jsonl_for_pii(path)
        if hits:
            rel = path.relative_to(REPO_ROOT)
            for line_no, kind, sample in hits[:3]:
                all_hits.append(f"{rel}:{line_no}: {kind} = {sample!r}")
            if len(hits) > 3:
                all_hits.append(f"{rel}: ...and {len(hits) - 3} more matches")

    assert not all_hits, "PII detected in committed corpus:\n" + "\n".join(all_hits)


# ---------------------------------------------------------------------------
# Sanity: make sure our parsers actually find data (catches relocations).
# ---------------------------------------------------------------------------


def test_gate_helpers_find_their_inputs() -> None:
    """If someone moves a directory, the gates above could silently pass on
    empty inputs. This test exists to make that loud."""
    assert _yaml_recipe_slugs(), "no YAML recipes discovered"
    assert _sources_slugs(), "no SOURCES.md slugs parsed — regex out of date?"
    assert _committed_jsonl([REPO_ROOT / "data" / "eval"]), "no eval JSONL discovered"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
