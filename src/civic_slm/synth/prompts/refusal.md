You are generating refusal training data — adversarial questions where the
correct behavior is "I don't know based on the provided context."

# Source chunk

Jurisdiction: {jurisdiction}, {state}
Document type: {doc_type}

```
{chunk_text}
```

# Task

Emit exactly {n} JSON objects, one per line. Each object must have:

```
{{"task": "refusal",
  "system": "You are a civic assistant. Answer using ONLY the provided context. If the answer isn't in the context, say so without guessing.",
  "input": "Context:\\n<copy of the chunk>\\n\\nQuestion: <plausible-sounding question whose answer is NOT in the chunk>",
  "output": "I don't have that information in the provided context. <Optional: suggest what document might have it.>"}}
```

# Quality bar

- The question must be PLAUSIBLE — sound like a real question a resident might ask about this kind of document — but the chunk MUST NOT contain the answer.
- The output must clearly decline. No partial guesses or "maybe it's...".
- Vary the type of withheld information: dates, names, dollar amounts, vote counts, procedural steps.
- Output ONLY the {n} JSON lines.
