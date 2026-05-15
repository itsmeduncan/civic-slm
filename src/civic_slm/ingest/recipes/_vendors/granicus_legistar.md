Open {start_url} (a Granicus archive page or a Legistar Calendar.aspx
page — common for large U.S. cities such as Los Angeles, New York,
Chicago, and Boston).

Locate the regular City Council (or equivalent legislative body)
meetings going back to {since}, up to {max_docs} meetings. On Granicus
sites the agenda is often a link near each meeting row; on Legistar it
is typically reachable from the meeting Detail page under "Agenda".
Resolved URLs commonly look like `/View.ashx?M=A&ID=...&GUID=...` on
Legistar or `/agenda_publish.cfm?...` on Granicus. Verify each URL
resolves to a PDF before returning it.

Skip workshops, study sessions, committee-only meetings, and closed
sessions. Only return regular legislative-body agendas.

For each meeting, return:

- title: the meeting name
- meeting_date: the date in YYYY-MM-DD format
- source_url: the direct URL to the agenda PDF

Return strictly as a JSON array of objects with the three keys above.
No commentary, no markdown, no preamble.

Note for future work: Legistar exposes a REST endpoint at
`https://webapi.legistar.com/v1/{{jurisdiction}}/Events` that returns
the same data without a browser. A non-browser recipe variant can
subclass `YamlRecipe` and override `discover` to use HTTP directly.
