"""
Lock-in acquisition app: SmarAct MCS2 stage + Stanford Research SR865A.

Step-scans the MCS2 translation stage and reads the lock-in's demodulated
signal at every commanded position, plotting the trace live against position.
It is the imaging app (`main.py`) with the camera replaced by the lock-in: same
panel/driver/scan structure, no camera worker process to start -- both
instruments are polled straight from the GUI process, and the scan itself runs
on one worker thread.

    python main_lockin.py                 # real hardware
    python main_lockin.py --simulate       # no hardware; fake stage + fake signal
"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from ui.lockin_scan_panel import DEFAULT_SAVE_DIR
from ui.lockin_window import LockInMainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCS2 stage + SR865A lock-in step-scan acquisition")
    parser.add_argument(
        "--simulate", action="store_true",
        help="Preselect the simulated stage and lock-in (no hardware needed)")
    parser.add_argument(
        "--save-dir", default=DEFAULT_SAVE_DIR,
        help=f"Default folder for saved scans (default: {DEFAULT_SAVE_DIR})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = LockInMainWindow(save_dir=args.save_dir, simulate=args.simulate)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
