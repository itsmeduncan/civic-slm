Open {start_url} (IQM2 / iCompass Calendar.aspx page; common for
mid-size California cities like Santa Monica).

Locate the calendar grid and identify the most recent regular City
Council meetings going back to {since}, up to {max_docs} meetings.
Each meeting links to a `Detail_Meeting.aspx?ID=...` page; on that
page locate the "Agenda Packet" or "Agenda" download — those URLs
typically follow `/Citizens/FileOpen.aspx?Type=14&ID=...` or end in
`.pdf`. Verify the resolved URL is a PDF before returning it.

Skip workshops, special meetings, study sessions, and closed sessions.
Only return regular City Council meeting agendas.

For each meeting, return:

- title: the meeting name (e.g. "Regular City Council Meeting")
- meeting_date: the date in YYYY-MM-DD format
- source_url: the direct URL to the agenda PDF

Return strictly as a JSON array of objects with the three keys above.
No commentary, no markdown, no preamble.
