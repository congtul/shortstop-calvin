"""Check that the third-party assets are where the patched MDT config expects
them, and that they are the files every reported run used (SHA-256).

    python scripts/check_assets.py
"""
import hashlib
import sys
from pathlib import Path

MDT = Path(__file__).resolve().parents[1] / "mdt_policy"
ASSETS = [
    ("MDT checkpoint mdtv-1-abcd",
     MDT / "pretrained_models/CALVIN_ABCD/mdtv_1_abcd/saved_models/avg_seq_len=4.64.ckpt",
     "2cb1ced81719b39c6f761ad578fce0c17d87ceba2b1a8d441dc96474a1c4bd6d"),
    ("MDT checkpoint config", MDT / "pretrained_models/CALVIN_ABCD/mdtv_1_abcd/.hydra/config.yaml", None),
    ("CALVIN debug dataset", MDT / "dataset/calvin_debug_dataset/validation/.hydra/merged_config.yaml", None),
    ("Voltron v-cond encoder (downloaded on first model load)", MDT / "vcache/v-cond/v-cond.pt",
     "de6c68ab930291bc655b56d692f6880919c497180022b6c9c28867b0c0007fcf"),
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ok = True
    for name, path, digest in ASSETS:
        if not path.exists():
            print(f"MISSING   {name}: {path}")
            ok = False
        elif digest is not None and _sha256(path) != digest:
            print(f"DIFFERENT {name}: {path} (sha256 does not match the file the runs used)")
            ok = False
        else:
            print(f"ok        {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
