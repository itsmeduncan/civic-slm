"""Ask a grounded civic question of a locally-served model and print the answer.

Prerequisites:
- `uv sync` (no extras needed)
- An OpenAI-compatible server reachable at $CIVIC_SLM_CANDIDATE_URL
  (defaults to http://127.0.0.1:8080) serving $CIVIC_SLM_CANDIDATE_MODEL
  (defaults to mlx-community/Qwen2.5-7B-Instruct-4bit).

Quick start:

    # In one terminal:
    uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --port 8080

    # In another:
    uv run python examples/01_ask_a_question.py

This is the smallest possible "what does civic-slm do?" demo. It uses the
same `ChatClient` and the same factuality system prompt the eval runner
uses, on a hand-typed staff-report excerpt. No data is written, no API
spend.
"""

from __future__ import annotations

from civic_slm.serve import runtimes
from civic_slm.serve.client import ChatClient

CONTEXT = """\
STAFF REPORT — Planning Commission
File: CUP 24-031
Applicant: Coastal Surf School, LLC
Location: 415 N. El Camino Real
Request: Authorize commercial surf instruction operations on weekday afternoons.
Recommendation: Approve subject to conditions.
Fiscal Impact: None.
"""

QUESTION = "Who is the applicant on CUP 24-031, and what is the staff recommendation?"

SYSTEM_PROMPT = (
    "You are a civic assistant. Answer the user's question using ONLY the provided "
    "context. Cite specific section names or item numbers from the context. If the "
    "answer is not in the context, say you don't know."
)


def main() -> None:
    client = ChatClient(base_url=runtimes.candidate_url(), model=runtimes.candidate_model())
    user = f"Context:\n{CONTEXT}\nQuestion: {QUESTION}"
    print(f"→ asking {client.model} at {client.base_url}\n")
    resp = client.chat(SYSTEM_PROMPT, user)
    print(resp.text)
    print(f"\n({resp.latency_ms:.0f} ms)")


if __name__ == "__main__":
    main()
