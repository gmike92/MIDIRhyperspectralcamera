"""Shared Qt wrapper for a blocking hardware driver.

Same contract as `ui.stages.StageController`, kept separate so the lock-in
acquisition app does not depend on the camera app's stage panel: run blocking
calls (connect, move, reference) on a worker thread, report status/position
through signals, and expose a `busy` flag the panels use to grey out controls.
"""
from __future__ import annotations

import threading

from PyQt6 import QtCore


class DeviceController(QtCore.QObject):
    """Wraps a driver; runs blocking ops on a thread, emits updates."""

    status = QtCore.pyqtSignal(str)
    reading = QtCore.pyqtSignal(str)
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(self, driver, fmt=None) -> None:
        super().__init__()
        self.driver = driver
        self._fmt = fmt                 # driver -> str, for the `reading` signal
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def connected(self) -> bool:
        return bool(getattr(self.driver, "is_connected", False))

    def run(self, fn, done_msg: str = "") -> bool:
        """Run `fn` on a worker thread. False if one is already running."""
        if self._busy:
            return False
        self._busy = True
        self.busy_changed.emit(True)

        def _work():
            try:
                fn()
                if done_msg:
                    self.status.emit(done_msg)
            except Exception as e:  # noqa: BLE001
                self.status.emit(f"error: {e}")
            finally:
                self._busy = False
                self.busy_changed.emit(False)
                self._emit_reading()

        threading.Thread(target=_work, daemon=True).start()
        return True

    def poll(self) -> None:
        if not self._busy:
            self._emit_reading()

    def _emit_reading(self) -> None:
        if self._fmt is None:
            return
        try:
            if self.connected:
                self.reading.emit(self._fmt(self.driver))
        except Exception:  # noqa: BLE001
            pass
