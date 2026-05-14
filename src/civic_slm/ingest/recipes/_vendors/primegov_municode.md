Open {start_url}. This is the long-tail vendor template covering
PrimeGov (`/Public/Calendar` style) and Municode-published meeting
indexes. **Treat this as experimental** — the agent will need to be
patient with these sites; they are inconsistent across jurisdictions.

Locate the regular City Council (or equivalent legislative body)
meetings going back to {since}, up to {max_docs} meetings. PrimeGov
Detail pages typically expose an "Agenda" link; Municode meeting pages
may link directly to a hosted PDF or to an external Granicus/CivicPlus
archive. If the source URL resolves to anything other than a PDF
(HTML index, login wall, error page) **skip it** rather than guessing.

Skip workshops, study sessions, committee-only meetings, and closed
sessions. Only return regular legislative-body agendas.

For each meeting, return:

- title: the meeting name
- meeting_date: the date in YYYY-MM-DD format
- source_url: the direct URL to the agenda PDF

Return strictly as a JSON array of objects with the three keys above.
No commentary, no markdown, no preamble.

If the agent cannot find any meetings (or all candidate URLs resolve
to non-PDF pages) return `[]`. The maintainer will see the empty
manifest and can either fall back to a Python recipe or open a
follow-up issue describing the vendor quirk.
