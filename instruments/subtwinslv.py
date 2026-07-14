"""
subtwinslv.py -- TWINS interferometer scan + FTIR spectrum.

System 2 of the setup: a NIREOS TWINS common-path interferometer. We scan its
internal wedge stage (twins_stage.TwinsStage) and, at each position, read a
scalar from the camera (ROI mean of a frame) to build an interferogram, then
DFT it into a spectrum using a NIREOS calibration file.

Ported from the LabVIEW pump-probe code (sub_twins_lw.py) but DECOUPLED from
LabVIEW: instead of triggering Experiment_manager.vi it takes any
`frame_source` callable returning a 2-D numpy frame -- e.g. our IRC806 camera's
`get_frame`. The interferogram->spectrum math is the VERBATIM repo
`SpectrumProcessor` (instruments/spectrum_processor.py, pandas+scipy). Runs a
simulated scan with no hardware via `simulate=True`.

Usage (with the real camera, later):
    from instruments.twins_stage import TwinsStage
    from instruments.subtwinslv import TwinsScanner
    stage = TwinsStage(); stage.connect()
    scanner = TwinsScanner(stage, frame_source=camera.get_frame,
                           calibration_file=r"...\\parameters_cal.txt")
    pos, ifg = scanner.scan(23.8, 24.8, 120, roi=(200, 320, 250, 400))
    wl, spec = scanner.compute_spectrum(wl_start=8.0, wl_stop=14.0)
    scanner.save("twins_scan")

Usage (now, no hardware):
    python subtwinslv.py            # runs a simulated scan + spectrum
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # package import (run from gui/) vs standalone (run from instruments/)
    from instruments.spectrum_processor import SpectrumProcessor
except ImportError:  # pragma: no cover
    from spectrum_processor import SpectrumProcessor

FrameSource = Callable[[], Optional[np.ndarray]]
Roi = Optional[tuple]  # (r0, r1, c0, c1) or None for full-frame mean


def bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Block-average a 2-D image by an integer factor (NxN super-pixels).

    Crops to a multiple of `factor` then averages each NxN block. factor<=1 is
    a no-op. Used to bin the ROI before building the hyperspectral datacube
    (better SNR, smaller cube)."""
    if factor is None or factor <= 1:
        return img
    h, w = img.shape
    bh, bw = h // factor, w // factor
    if bh == 0 or bw == 0:
        return img
    cropped = img[:bh * factor, :bw * factor]
    return cropped.reshape(bh, factor, bw, factor).mean(axis=(1, 3))


# ===========================================================================
# TWINS scanner -- drives the stage + reads scalars from a frame source
# ===========================================================================
class TwinsScanner:
    def __init__(self, stage, frame_source: FrameSource,
                 calibration_file: Optional[str] = None) -> None:
        self.stage = stage
        self.frame_source = frame_source
        self.processor = SpectrumProcessor(calibration_file)
        self.positions = None
        self.interferogram = None

    @staticmethod
    def _roi_mean(frame: np.ndarray, roi: Roi) -> float:
        if frame is None:
            return float("nan")
        if roi is None:
            return float(np.mean(frame))
        r0, r1, c0, c1 = roi
        return float(np.mean(frame[r0:r1, c0:c1]))

    def _read_scalar(self, roi: Roi, frames_avg: int,
                     discard: int = 1, timeout_s: float = 2.0) -> float:
        """Average `frames_avg` FRESH ROI means at the current wedge position.

        Like _read_roi_slice, only DISTINCT camera frames are averaged (a slow
        camera never gets the same buffered frame counted twice) and the first
        `discard` new frames after the move are dropped as possibly-in-flight.
        Falls back to whatever frame is available if the stream stalls.
        """
        vals = []
        last = self.frame_source()           # pre-settle frame: wait for it to change
        deadline = time.time() + timeout_s
        dropped = 0
        while len(vals) < max(1, frames_avg):
            frame = self.frame_source()
            if frame is None or frame is last:
                if time.time() > deadline:
                    break
                time.sleep(0.002)
                continue
            last = frame
            deadline = time.time() + timeout_s
            if dropped < discard:            # skip a possibly in-flight frame
                dropped += 1
                continue
            vals.append(self._roi_mean(frame, roi))
        if not vals:                         # stream stalled -> use latest frame
            vals.append(self._roi_mean(self.frame_source(), roi))
        return float(np.nanmean(vals))

    def scan(self, start_mm: float, stop_mm: float, n_steps: int,
             roi: Roi = None, settle_s: float = 0.05, frames_avg: int = 1,
             progress: Optional[Callable[[int, int, float, float], None]] = None,
             should_abort: Optional[Callable[[], bool]] = None):
        """Move the TWINS stage across [start, stop] and build the interferogram.

        Step size = abs(stop-start)/(n_steps-1). Returns (positions_mm,
        interferogram). `progress(i, n, pos, value)` is called per point if
        supplied; `should_abort()` is polled per point to allow a clean stop
        (returns the partial scan truncated to the points taken).
        """
        targets = np.linspace(start_mm, stop_mm, int(n_steps))
        positions = np.zeros(n_steps)
        interferogram = np.zeros(n_steps)
        taken = 0
        for i, target in enumerate(targets):
            if should_abort is not None and should_abort():
                break
            self.stage.move_to(float(target))
            self.stage.wait_for_stop()
            if settle_s:
                time.sleep(settle_s)
            positions[i] = self.stage.get_position()
            interferogram[i] = self._read_scalar(roi, frames_avg)
            taken = i + 1
            if progress:
                progress(taken, n_steps, positions[i], interferogram[i])
        positions = positions[:taken]
        interferogram = interferogram[:taken]
        self.positions = positions
        self.interferogram = interferogram
        self.processor.set_data(positions, interferogram)
        return positions, interferogram

    def _read_roi_slice(self, roi: Roi, frames_avg: int, bin_factor: int = 1,
                        discard: int = 1, timeout_s: float = 2.0, background=None):
        """Average `frames_avg` FRESH live frames at the current wedge position.

        Only distinct camera frames (a new array object each time the camera
        publishes) are averaged, so a slow camera never gets the same buffered
        frame counted twice. The first `discard` new frames after the move are
        dropped because they may have been captured/in flight DURING the wedge
        move -- this is the step-scan freshness guarantee. `timeout_s` bounds the
        wait per frame so a stalled stream can't hang the scan.
        """
        acc = None
        count = 0
        dropped = 0
        last = self.frame_source()          # pre-settle frame: wait for it to change
        deadline = time.time() + timeout_s
        while count < max(1, frames_avg):
            frame = self.frame_source()
            if frame is None or frame is last:
                if time.time() > deadline:
                    break
                time.sleep(0.002)
                continue
            last = frame
            deadline = time.time() + timeout_s
            if dropped < discard:           # skip a possibly in-flight frame
                dropped += 1
                continue
            sl = np.asarray(frame, dtype=np.float32)
            # Subtract the captured background (same ROI crop) BEFORE binning, so
            # the interferogram is background-corrected fixed-pattern-wise.
            if background is not None and background.shape == sl.shape:
                sl = sl - np.asarray(background, dtype=np.float32)
            if roi is not None:
                sl = sl[roi[0]:roi[1], roi[2]:roi[3]]
            sl = bin_image(sl, bin_factor)
            acc = sl.copy() if acc is None else acc + sl
            count += 1
        return None if acc is None else acc / count

    def _wait_for_stream(self, should_abort=None, status_cb=None,
                         timeout_s: float = 60.0, need: int = 5,
                         settle_after: float = 0.5) -> bool:
        """Block until the live stream resumes after a camera freeze.

        Waits for `need` consecutive FRESH (distinct) frames, then a settle. The
        camera process auto-reconnects on a link drop and restores the same
        integration time + NUC/correction (so frames match those before the
        freeze) -- here we just wait it out. Returns True once streaming is back,
        False on abort or `timeout_s`.
        """
        if status_cb:
            status_cb("camera not streaming -- waiting for recovery...")
        t0 = time.time()
        last = self.frame_source()
        fresh = 0
        while time.time() - t0 < timeout_s:
            if should_abort is not None and should_abort():
                return False
            frame = self.frame_source()
            if frame is not None and frame is not last:
                last = frame
                fresh += 1
                if fresh >= need:
                    time.sleep(settle_after)
                    if status_cb:
                        status_cb("camera recovered -- resuming scan")
                    return True
            else:
                time.sleep(0.05)
        if status_cb:
            status_cb("camera did not recover (timeout) -- stopping scan")
        return False

    def scan_cube(self, start_mm: float, stop_mm: float, n_steps: int,
                  roi: Roi, settle_s: float = 0.05, frames_avg: int = 1,
                  bin_factor: int = 1, discard_frames: int = 1, background=None,
                  progress: Optional[Callable[..., None]] = None,
                  should_abort: Optional[Callable[[], bool]] = None,
                  status_cb: Optional[Callable[[str], None]] = None):
        """Scan the TWINS wedge storing the full 2-D ROI at each position.

        Step-scan: move -> wait_for_stop -> settle -> average `frames_avg` FRESH
        frames (see _read_roi_slice). Returns (positions_mm, datacube) with shape
        (n_taken, h, w). `progress(i, n, pos, value)` is called per point, where
        value is the ROI-mean (for a live interferogram preview).
        """
        targets = np.linspace(start_mm, stop_mm, int(n_steps))
        positions = []
        cube = []
        for i, target in enumerate(targets):
            if should_abort is not None and should_abort():
                break
            self.stage.move_to(float(target))
            self.stage.wait_for_stop()
            if settle_s:
                time.sleep(settle_s)
            pos = self.stage.get_position()
            sl = self._read_roi_slice(roi, frames_avg, bin_factor, discard_frames,
                                      background=background)
            if sl is None:
                # Camera froze: the wedge has NOT moved, so wait for the stream to
                # come back (same exposure + NUC restored on reconnect) and then
                # re-acquire THIS position so the interferogram has no gap.
                if not self._wait_for_stream(should_abort, status_cb):
                    break
                if settle_s:
                    time.sleep(settle_s)
                pos = self.stage.get_position()
                sl = self._read_roi_slice(roi, frames_avg, bin_factor, discard_frames,
                                          background=background)
                if sl is None:
                    continue
            positions.append(pos)
            cube.append(sl)
            if progress:
                progress(len(cube), int(n_steps), pos, float(np.mean(sl)))
        self.cube_positions = np.asarray(positions)
        self.datacube = np.asarray(cube) if cube else None
        return self.cube_positions, self.datacube

    def compute_spectrum(self, **kwargs):
        return self.processor.compute_spectrum(**kwargs)

    def save(self, basename: str = "twins_scan", out_dir: str = "scan_data") -> str:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = str(Path(out_dir) / f"{basename}_{stamp}")
        with open(stem + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["position_mm", "interferogram"])
            for p, v in zip(self.positions, self.interferogram):
                w.writerow([p, v])
        np.savez(stem + ".npz",
                 positions=self.positions, interferogram=self.interferogram,
                 wavelengths=self.processor.wavelengths, spectrum=self.processor.spectrum)
        print(f"[TWINS] saved {stem}.csv / .npz")
        return stem


# ===========================================================================
# Simulated demo (no hardware): synthetic two-line MWIR source
# ===========================================================================
def _simulated_demo():
    from twins_stage import TwinsStage

    stage = TwinsStage()
    stage.connect(simulate=True, home=False)

    # Synthetic interferogram: sum of two cosines vs wedge position (mm).
    # Frame source returns a uniform frame whose mean encodes the signal at the
    # current stage position (mocks "camera ROI mean").
    def frame_source():
        x = stage.get_position()
        x0 = 24.3
        sig = (np.cos(2 * np.pi * 0.12 * (x - x0) * 1000)
               + 0.6 * np.cos(2 * np.pi * 0.16 * (x - x0) * 1000))
        envelope = np.exp(-((x - x0) / 0.25) ** 2)
        value = 1000.0 + 400.0 * sig * envelope + np.random.normal(0, 5)
        return np.full((32, 32), value, dtype=np.float32)

    scanner = TwinsScanner(stage, frame_source)
    pos, ifg = scanner.scan(23.8, 24.8, 200,
                            progress=lambda i, n, p, v: None)
    wl, spec = scanner.compute_spectrum(wl_start=4.0, wl_stop=12.0)
    print(f"[demo] scanned {len(pos)} pts; interferogram "
          f"min/max {ifg.min():.0f}/{ifg.max():.0f}")
    if spec is not None:
        peak = wl[int(np.argmax(spec))]
        print(f"[demo] spectrum computed ({len(spec)} pts); peak ~{peak:.2f} µm")
    stage.disconnect()


if __name__ == "__main__":
    _simulated_demo()
