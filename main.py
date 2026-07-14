from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from multiprocessing import shared_memory

from PyQt6.QtWidgets import QApplication
from loguru import logger

from ui.main_window import MainWindow
from worker_camera import camera_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IRC806 MWIR live viewer")
    parser.add_argument(
        "--mode",
        choices=("auto", "irc806", "mock"),
        default="irc806",
        help="Camera backend (irc806 = IRCameras MWIR via Pleora eBUS)",
    )
    parser.add_argument("--fps", type=float, default=120.0,
                        help="Max acquisition rate (cap); camera native is 120 Hz")
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = parse_args()

    frame_queue = mp.Queue(maxsize=4)
    control_queue = mp.Queue()
    frame_buffer_shape = (4096, 4096)
    frame_buffer = shared_memory.SharedMemory(
        create=True, size=frame_buffer_shape[0] * frame_buffer_shape[1] * 2
    )
    worker_config = {
        "mode": args.mode,
        "target_fps": args.fps,
        "shared_frame_name": frame_buffer.name,
        "shared_frame_shape": frame_buffer_shape,
    }

    process = mp.Process(target=camera_worker, args=(frame_queue, control_queue, worker_config))
    process.start()
    logger.info(f"Camera worker started with PID {process.pid}")

    app = QApplication(sys.argv)
    window = MainWindow(
        frame_queue,
        control_queue,
        initial_mode=args.mode,
        shared_frame_name=frame_buffer.name,
        shared_frame_shape=frame_buffer_shape,
    )
    window.show()

    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        control_queue.put({"type": "stop"})
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            logger.warning("Worker did not exit in time and was terminated")
        frame_buffer.close()
        try:
            frame_buffer.unlink()
        except FileNotFoundError:
            pass
        sys.exit(exit_code)
