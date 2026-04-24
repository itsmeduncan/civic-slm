You are generating structured extraction training data: given a civic document
fragment, output a JSON object with the requested fields.

# Source chunk

Jurisdiction: {jurisdiction}, {state}
Document type: {doc_type}

```
{chunk_text}
```

# Task

Emit exactly {n} JSON objects, one per line. Each object must have:

```
{{"task": "extract",
  "system": "You are a civic data extractor. Output ONLY a JSON object with the requested fields, no prose.",
  "input": "Schema: <name>\\n\\nDocument:\\n<copy of chunk>\\n\\nReturn JSON.",
  "output": "<a JSON object with fields drawn ONLY from the chunk; null if the field isn't present>"}}
```

# Schemas to choose from

- `staff_report`: file_number, applicant, location, recommendation, fiscal_impact
- `agenda_item`: item_number, title, recommended_action, requestor
- `contract`: vendor, term_years, amount_not_to_exceed, scope
- `appeal_notice`: case_number, decision_date, appeal_deadline, fee, where_to_file
- `ordinance`: ordinance_number, title, sections_amended, effective_date
- `resolution`: resolution_number, title, sponsor, vote_required
- `budget_item`: program, fund, fiscal_year, amount

# Quality bar

- Pick a schema that fits the chunk. Don't force it.
- Use null for fields the chunk doesn't mention. Never invent values.
- Output ONLY the {n} JSON lines.
