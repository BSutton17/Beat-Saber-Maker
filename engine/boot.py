"""Import this first: puts the copied engine on the path and swaps its MATLAB-based decoder for audio_io."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE / "core", HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
# this folder's mapfiles must win over any other
sys.modules.pop("mapfiles", None)

import audio_io  # noqa: E402
import metrics  # noqa: E402

metrics.decode = audio_io.decode
