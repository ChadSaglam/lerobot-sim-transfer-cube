import sys
from pathlib import Path

# The analysis scripts live in scripts/ and are not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
