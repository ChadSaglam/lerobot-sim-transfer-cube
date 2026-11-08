"""Start a long-running command fully detached from the terminal.

`nohup ... &` is not enough on macOS: closing the Terminal window sends SIGTERM to every
process still attached to that window's tty. This launcher starts the command in a new
session (no controlling terminal), with stdin from /dev/null and stdout/stderr appended to a
log file, wrapped in `caffeinate -is` so the Mac does not sleep while it runs.

Usage:
    python3 scripts/run_detached.py bash scripts/overnight_queue.sh
    python3 scripts/run_detached.py bash scripts/extend_seed1000.sh

Stop it later with:  pkill -f overnight_queue.sh  (and pkill -f lerobot-eval / lerobot-train)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(__file__).resolve().parents[1]
    cmd = sys.argv[1:]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log = root / "outputs" / "train" / f"detached_{Path(cmd[-1]).stem}_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        proc = subprocess.Popen(
            ["caffeinate", "-is", *cmd],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"started pid {proc.pid} (detached)")
    print(f"log: {log.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
