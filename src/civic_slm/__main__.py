"""Enable `python -m civic_slm` as an equivalent to the `civic-slm` script.

The jurisdiction pipeline composer launches sub-stages via `python -m civic_slm`
so each stage runs in a clean process; without this entry point that
subprocess call dies with "No module named civic_slm.__main__".
"""

from __future__ import annotations

from civic_slm.cli import app

if __name__ == "__main__":
    app()
