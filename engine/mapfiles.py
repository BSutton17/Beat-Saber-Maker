"""This program's stand-in for AI Mapper's mapfiles.py: the copied engine finds its models, song list and stems in this engine folder, and writes maps to the chosen output folder."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
# where finished maps go: ChroMapper's CustomWIPLevels folder, set by the app
WIP = Path(os.environ.get("BSM_OUTPUT_DIR", PROJECT.parent / "output"))
