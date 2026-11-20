"""Invoke the repository's official pair-recovery CLI without installing a new package."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PHAROS" / "src"))

from pharos_cell.evaluation.pair_recovery_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
