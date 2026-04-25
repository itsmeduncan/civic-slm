# Examples — first 5 minutes

These scripts are runnable copy-paste, no flags-required, demos for
people who land on the repo and want to see what civic-slm actually does
before committing to read all of `docs/USAGE.md`.

| Script                      | What it shows                                     | Runs without               |
| --------------------------- | ------------------------------------------------- | -------------------------- |
| `01_ask_a_question.py`      | Ask a grounded civic question + cite the answer   | training, ingest, internet |
| `02_run_factuality_eval.py` | Run the shipped factuality benchmark end-to-end   | training, ingest           |
| `03_inspect_a_baseline.py`  | Read the committed base-Qwen baseline + summarize | a server                   |

Each script declares its own prerequisites at the top in a docstring.
The default examples need only `uv sync` and a running OpenAI-compatible
server (e.g., `mlx_lm.server` or `llama-server`). They never talk to
Anthropic and never write to `data/`.

## Running them

```bash
# In one terminal, start a local OpenAI-compatible server. Any will do;
# the scripts read CIVIC_SLM_CANDIDATE_URL / CIVIC_SLM_CANDIDATE_MODEL
# the same way the eval runner does.
uv run mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080

# In another terminal:
uv run python examples/01_ask_a_question.py
```

If you don't have a local server yet, `examples/03_inspect_a_baseline.py`
runs without one — it reads the committed baseline JSON and prints a
human summary, so it works the moment `uv sync` finishes.
