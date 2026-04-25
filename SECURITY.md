# Security Policy

## Supported versions

`civic-slm` is pre-1.0. Only the latest tagged release receives security fixes. There is no LTS.

## Reporting a vulnerability

**Do not open a public GitHub issue for security problems.**

Use one of:

1. GitHub's private vulnerability reporting: https://github.com/itsmeduncan/civic-slm/security/advisories/new
2. Email: itsmeduncan@gmail.com with subject prefix `[civic-slm security]`. PGP on request.

Please include:

- A description of the issue and its impact.
- Steps to reproduce, or a minimal proof-of-concept.
- The commit SHA or release tag affected.
- Whether you intend to disclose publicly, and on what timeline.

## Response timeline

| Stage                         | Target              |
| ----------------------------- | ------------------- |
| Acknowledgement of report     | within 3 days       |
| Initial assessment + severity | within 7 days       |
| Fix or mitigation in main     | within 30 days      |
| Coordinated public disclosure | by mutual agreement |

Critical issues affecting model integrity, training-data poisoning, or RCE in
the ingestion pipeline get prioritized over the schedule above.

## Threat model — what is in scope

- The Python package on PyPI (`civic-slm`) and any code in this repo.
- Released model weights and adapters (HF Hub artifacts).
- The synthetic-data pipeline (`src/civic_slm/synth/`) and the ingestion pipeline (`src/civic_slm/ingest/`), including the LLM-driven crawler.
- The OpenAI-compatible serving glue in `src/civic_slm/serve/`.

### Known dual-use surfaces (documented, not vulnerabilities)

- **Prompt injection in training data.** `src/civic_slm/synth/generate.py` interpolates raw civic-document text into a Claude prompt. A malicious or compromised civic site can attempt to inject instructions that affect generated SFT examples. Mitigations: human review of the first 500 examples (`scripts/review_sft.py`), Pydantic schema validation (`src/civic_slm/schema.py`), and the planned content-source allowlist per recipe.
- **LLM-driven crawler agency.** `src/civic_slm/ingest/recipes/_browser.py` runs a `browser-use` agent. The agent is constrained per recipe but can in principle follow links the recipe author did not anticipate. Forks should pin recipes to known domains.
- **Strict-local mode.** Setting `CIVIC_SLM_STRICT_LOCAL=1` makes synth, judge, and crawler refuse to call Anthropic. This is enforced at runtime (`src/civic_slm/llm/backend.py:120-128`, `src/civic_slm/ingest/recipes/_browser.py:50-58`), but it does **not** prevent Hugging Face model downloads or other local network calls.

## What is out of scope

- Bugs that require a malicious local user with shell access to your training machine.
- Issues in transitive dependencies that have an upstream fix; please report those upstream and we will pin.
- Model-quality issues, hallucinations, or refusal behavior — these are governance/eval concerns tracked in the model card.

## Disclosure credits

We credit reporters in `CHANGELOG.md` and the security advisory unless you ask us not to.
