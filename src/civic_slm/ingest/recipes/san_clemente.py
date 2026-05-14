"""San Clemente, CA — demo recipe.

Used as the reference recipe. Copy `_template.py` (not this file) to add a new
jurisdiction — that file has a cleaner commented skeleton.

For new jurisdictions, the YAML path (see `civic-slm new-recipe`) covers the
~80% case; this Python recipe is the escape-hatch reference, kept around to
document what a recipe with both PDF (`discover`) and video (`discover_videos`)
support looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

from civic_slm.ingest.harness import DiscoveredDoc, DiscoveredVideo
from civic_slm.ingest.recipes._browser import run_browser_agent
from civic_slm.ingest.recipes._youtube import youtube_channel_videos

INSTRUCTION = """\
Open https://www.san-clemente.org/ and navigate to the City Council meetings page.
Find the list of past City Council meetings going back to {since}, up to {max_docs} most recent.
For each meeting, return:
  - title: the meeting name (e.g. "City Council Meeting")
  - meeting_date: the date in YYYY-MM-DD format
  - source_url: the direct URL to the PDF agenda for that meeting
Skip workshops, special meetings, and closed sessions. Only return regular City Council agendas.
Return strictly as a JSON array of objects with the three keys above.
"""

# San Clemente streams its council meetings on this channel. The handle is
# referenced verbatim in `docs/SOURCES.md` under "san-clemente (CA)" — keep
# them in sync. yt-dlp enumerates this URL without a browser; captions
# (when published) are preferred over Whisper ASR.
YOUTUBE_CHANNEL = "https://www.youtube.com/@san-clemente-tv/videos"


@dataclass(frozen=True)
class SanClementeRecipe:
    jurisdiction: str = "san-clemente"
    state: str = "CA"
    instruction: str = INSTRUCTION
    youtube_channel: str = YOUTUBE_CHANNEL

    async def discover(self, *, since: str, max_docs: int) -> list[DiscoveredDoc]:
        return await run_browser_agent(self.instruction, since=since, max_docs=max_docs)

    async def discover_videos(self, *, since: str, max_videos: int) -> list[DiscoveredVideo]:
        """Enumerate San Clemente TV's council recordings via yt-dlp.

        `youtube_channel_videos()` is sync; calling it from this async
        method is fine because yt-dlp's metadata pass doesn't block the
        loop meaningfully and the orchestrator (`harness.crawl_videos`)
        is also sync once it hits the actual download.
        """
        return youtube_channel_videos(self.youtube_channel, since=since, max_videos=max_videos)
