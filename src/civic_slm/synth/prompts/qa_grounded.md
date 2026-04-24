You are generating supervised fine-tuning data for a small language model that
helps Californians read and reason about their city's documents. Your job is to
read one chunk of a real civic document and emit grounded Q&A pairs.

# Source chunk

City: {city}
Document type: {doc_type}
Section path: {section_path}

```
{chunk_text}
```

# Task

Emit exactly {n} JSON objects, one per line (JSONL). Each object must have:

```
{{"task": "qa_grounded",
  "system": "You are a civic assistant. Answer using ONLY the provided context. Cite specific section names or item numbers. If the answer isn't in the context, say so.",
  "input": "Context:\\n<copy of the chunk>\\n\\nQuestion: <a question fully answerable from the chunk>",
  "output": "<concise, grounded answer that cites section/item numbers from the chunk>"}}
```

# Quality bar

- Questions must be answerable purely from the chunk. No outside knowledge.
- Answers must cite the section, item, or code reference verbatim from the chunk.
- Vary phrasing — formal, casual, "as a resident", etc. Don't repeat phrasing across the {n} examples.
- No hallucinated dates, dollar amounts, or names.
- Output ONLY the {n} JSON lines, no prose, no preamble, no fences.
