"""ShortStop on CALVIN. Importing the package sets SHORTSTOP_ROOT (the repo
root), which the patched mdt_policy/conf/mdt_evaluate.yaml uses for the
checkpoint, dataset and encoder-cache paths."""
import os
from pathlib import Path

os.environ.setdefault("SHORTSTOP_ROOT", str(Path(__file__).resolve().parents[2]))
