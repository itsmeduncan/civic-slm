Open {start_url} and locate the City Council meeting list (CivicPlus
AgendaCenter, typically under `/AgendaCenter` or
`/Citizens/AgendaCenter`).

Find the most recent regular City Council meetings going back to
{since}, up to {max_docs} meetings. For each meeting, the agenda PDF
link typically follows the pattern
`/AgendaCenter/ViewFile/Agenda/_MMDDYYYY-NNN`. Click through to confirm
the link resolves to a PDF before returning it.

Skip workshops, special meetings, and closed sessions. Skip past
meetings whose agenda link is missing or returns a non-PDF page. Only
return regular City Council agendas.

For each meeting, return:

- title: the meeting name (e.g. "Regular City Council Meeting")
- meeting_date: the date in YYYY-MM-DD format
- source_url: the direct URL to the agenda PDF

Return strictly as a JSON array of objects with the three keys above.
No commentary, no markdown, no preamble.
