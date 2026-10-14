"""
lockin_scan.py -- step-scan engine: SmarAct MCS2 stage + SR865A lock-in.

Sweeps the MCS2 stage across [start, stop] in N steps and, at each commanded
position, reads the demodulated signal from the lock-in. This is the piece that
"synchronises" the two instruments, and there is exactly ONE knob for it:

    move -> wait for the channel to stop moving     (hardware handshake)
         -> read the lock-in as fast as the link allows for `dwell_s`
         -> the point is the MEAN of everything read in that window

`dwell_s` is the whole story: longer dwell = better averaging and a slower scan.
The per-point error bar is the standard deviation of those same reads, and
`reads` records how many there were.

One caveat worth knowing rather than hiding: the window average includes the
lock-in's settling transient at the start, because the output is the input
through a low-pass filter and takes a few time constants to forget the previous
position. That bias is negligible while `dwell_s` is comfortably longer than the
time constant, and grows as the two approach each other -- so the scan reads tau
off the instrument and `dwell_warning()` says when the dwell is too short for
the filter actually configured.

Hardware-only (NO GUI): progress and abort are plain callbacks, so the same
scanner drives the Qt panel and a headless script.

Usage:
    from instruments.lockin_scan import LockInScanner
    sc = LockInScanner(stage, lockin)
    pos, val = sc.scan(-0.1, 0.1, 201, parameter="R", dwell_s=0.5)
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

#: Default waiting time at each scan position, in seconds.
DEFAULT_DWELL_S = 0.5

#: A dwell shorter than this many time constants is flagged: too much of the
#: averaging window is the settling transient from the previous position.
MIN_DWELL_TAU = 5.0

#: Parameters obtainable from a single `SNAP? X,Y,R`, so recording the
#: quadratures costs no extra query when the scanned channel is one of them.
_SNAP_COMPONENTS = {"X": 0, "Y": 1, "R": 2}

#: Floor on the "did the stage actually get there?" tolerance, in mm. The MCS2
#: closed-loop servo error is well under a micrometre, so 2 um is generous.
POSITION_TOLERANCE_FLOOR_MM = 0.002

#: ...and the tolerance also scales with the step, so coarse scans are not held
#: to a tighter standard than their own resolution.
POSITION_TOLERANCE_STEP_FRACTION = 0.2


def dwell_warning(dwell_s: float, time_constant_s: float) -> str:
    """'' if the dwell is long enough for this time constant, else why not."""
    if time_constant_s <= 0 or dwell_s >= MIN_DWELL_TAU * time_constant_s:
        return ""
    return (f"dwell {dwell_s:g} s is only {dwell_s / time_constant_s:.1f} x the "
            f"{time_constant_s:g} s time constant -- the average is biased by the "
            f"settling transient. Use at least {MIN_DWELL_TAU * time_constant_s:g} s, "
            f"or shorten the time constant on the lock-in.")


class LockInScanner:
    """Step-scan an MCS2 stage while reading an SR865A lock-in.

    After a scan the results stay on the instance (`positions`, `values`, ...)
    so the GUI can re-plot or save them without re-running anything.
    """

    def __init__(self, stage, lockin) -> None:
        self.stage = stage
        self.lockin = lockin
        self.positions = None       # mm, as READ BACK from the stage encoder
        self.targets = None         # mm, as commanded
        self.values = None          # mean of the selected parameter over the dwell
        self.errors = None          # std dev of the reads averaged into each point
        self.reads = None           # how many reads went into each point
        self.xyr = None             # (3, N) X, Y, R means, if record_xyr else None
        self.timestamps = None      # s since scan start
        self.parameter = "R"
        self.metadata = {}
        # Live view of a scan in progress: the full preallocated buffers plus how
        # many points have been filled. `live_trace()` reads these from the GUI
        # thread while the scan thread writes -- see its docstring.
        self.n_taken = 0
        self._buf_positions = None
        self._buf_values = None

    # -- one point -----------------------------------------------------------
    def _dwell_and_read(self, parameter: str, dwell_s: float, record_xyr: bool):
        """Read for `dwell_s` seconds and average everything.

        Returns (mean, std, n_reads, xyr_means or None). Always takes at least
        one read, however short the dwell.
        """
        lockin = self.lockin
        # When the scanned channel is X, Y or R, one SNAP? yields both the point
        # value and the quadratures -- half the traffic, and all three come from
        # the same instant.
        snap_only = record_xyr and parameter in _SNAP_COMPONENTS
        component = _SNAP_COMPONENTS.get(parameter, 0)

        vals: list[float] = []
        trios: list[list[float]] = []
        deadline = time.perf_counter() + max(0.0, float(dwell_s))
        while True:
            if snap_only:
                trio = lockin.snap("X", "Y", "R")
                trios.append(trio)
                vals.append(trio[component])
            else:
                vals.append(lockin.read_channel(parameter))
                if record_xyr:
                    trios.append(lockin.snap("X", "Y", "R"))
            if time.perf_counter() >= deadline:
                break

        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        xyr_means = None
        if trios:
            xyr_means = [sum(col) / len(col) for col in zip(*trios)]
        elif record_xyr:
            xyr_means = [float("nan")] * 3
        return mean, std, len(vals), xyr_means

    # -- the scan ------------------------------------------------------------
    def scan(self, start_mm: float, stop_mm: float, n_steps: int,
             parameter: str = "R",
             dwell_s: float = DEFAULT_DWELL_S,
             record_xyr: bool = True,
             move_timeout_s: float = 30.0,
             position_tolerance_mm: Optional[float] = None,
             progress: Optional[Callable[[int, int, float, float], None]] = None,
             should_abort: Optional[Callable[[], bool]] = None,
             status: Optional[Callable[[str], None]] = None):
        """Sweep [start_mm, stop_mm] in `n_steps`, averaging `parameter` for
        `dwell_s` at each position.

        Step size = abs(stop-start)/(n_steps-1). Returns (positions_mm, values),
        both truncated to the points actually taken if the scan was aborted.
        """
        lockin, stage = self.lockin, self.stage
        n = int(n_steps)
        targets = np.linspace(float(start_mm), float(stop_mm), n)
        dwell_s = max(0.0, float(dwell_s))
        step_mm = (abs(float(stop_mm) - float(start_mm)) / (n - 1)) if n > 1 else 0.0
        if position_tolerance_mm is None:
            position_tolerance_mm = max(POSITION_TOLERANCE_FLOOR_MM,
                                        POSITION_TOLERANCE_STEP_FRACTION * step_mm)

        # Pre-flight: a stage with no sensor or no reference cannot report where
        # it is, so every position in the scan would be fiction.
        if not getattr(stage, "has_sensor", True) and status:
            status("WARNING: stage reports no position sensor -- positions will "
                   "not be meaningful")
        if not getattr(stage, "is_referenced", True) and status:
            status("WARNING: stage is not referenced -- absolute positions are "
                   "meaningless until Find reference has run")

        # The lock-in's own filter settings are recorded with the scan (and used
        # to judge whether the dwell is long enough), read live so they describe
        # the instrument as it actually was, not as the GUI last remembered it.
        try:
            tau = float(lockin.time_constant)
        except Exception:  # noqa: BLE001
            tau = 0.0
        try:
            slope = int(lockin.filter_slope)
        except Exception:  # noqa: BLE001
            slope = 0

        self.parameter = parameter
        self.metadata = {
            "parameter": parameter,
            "start_mm": float(start_mm),
            "stop_mm": float(stop_mm),
            "n_steps": n,
            "step_um": (abs(float(stop_mm) - float(start_mm)) / (n - 1) * 1000.0
                        if n > 1 else 0.0),
            "dwell_s": dwell_s,
            "time_constant_s": tau,
            "filter_slope_dB_oct": slope,
            "dwell_per_tau": (dwell_s / tau) if tau > 0 else None,
            "record_xyr": bool(record_xyr),
            "position_tolerance_mm": float(position_tolerance_mm),
        }
        self.metadata.update(
            {f"lockin_{k}": v for k, v in lockin.settings_snapshot().items()})

        warning = dwell_warning(dwell_s, tau)
        self.metadata["dwell_warning"] = warning or None
        if status:
            status(warning or f"dwell {dwell_s:g} s per point (tau = {tau:g} s)")

        positions = np.full(n, np.nan)
        values = np.full(n, np.nan)
        errors = np.full(n, np.nan)
        reads = np.zeros(n, dtype=int)
        xyr = np.full((3, n), np.nan) if record_xyr else None
        stamps = np.full(n, np.nan)
        taken = 0
        t0 = time.time()
        # Publish the buffers before the first point so the GUI can draw the
        # trace as it fills, rather than only once the scan finishes.
        self._buf_positions, self._buf_values, self.n_taken = positions, values, 0

        for i, target in enumerate(targets):
            if should_abort is not None and should_abort():
                if status:
                    status("aborted")
                break

            if not stage.move_to(float(target)):
                if status:
                    status(f"move to {target:.5f} mm failed: "
                           f"{getattr(stage, 'last_error', '')}")
                break
            if not stage.wait_for_stop(timeout_s=move_timeout_s):
                if status:
                    status(f"stage did not settle at {target:.5f} mm: "
                           f"{getattr(stage, 'last_error', '')}")
                break

            positions[i] = stage.get_position()
            # Assert the stage actually arrived. MCS2 `Move` is documented NOT to
            # report an error when it cannot process the movement (wrong channel,
            # no positioner attached, range limit) -- it just silently does
            # nothing, `wait_for_stop` returns at once and the position never
            # changes. Without this check the scan would finish looking fine and
            # every point would carry the wrong position.
            error_mm = abs(positions[i] - float(target))
            if not (error_mm <= position_tolerance_mm):   # NaN-safe
                if status:
                    status(
                        f"stage did not reach {target:.5f} mm -- it reads "
                        f"{positions[i]:.5f} mm ({error_mm * 1000:.1f} µm off, "
                        f"tolerance {position_tolerance_mm * 1000:.1f} µm). "
                        "Check the channel index, that the positioner is "
                        "connected, that Find reference has run, and the range "
                        "limits.")
                break
            # Keep the simulator's fake signal tied to the real scan geometry.
            if getattr(lockin, "backend", None) == "sim":
                lockin.set_sim_position(positions[i])

            try:
                mean, std, n_reads, trio = self._dwell_and_read(
                    parameter, dwell_s, record_xyr)
            except Exception as e:  # noqa: BLE001
                if status:
                    status(f"lock-in read failed at {target:.5f} mm: {e}")
                break

            values[i] = mean
            errors[i] = std
            reads[i] = n_reads
            if xyr is not None and trio is not None:
                xyr[:, i] = trio
            stamps[i] = time.time() - t0
            taken = i + 1
            self.n_taken = taken        # after the writes: never publish a hole
            if progress:
                progress(taken, n, float(positions[i]), float(values[i]))

        self.targets = targets[:taken]
        self.positions = positions[:taken]
        self.values = values[:taken]
        self.errors = errors[:taken]
        self.reads = reads[:taken]
        self.xyr = xyr[:, :taken] if xyr is not None else None
        self.timestamps = stamps[:taken]
        self.metadata["points_taken"] = int(taken)
        self.metadata["duration_s"] = float(time.time() - t0)
        self.metadata["reads_per_point_mean"] = (
            float(np.mean(self.reads)) if taken else None)
        return self.positions, self.values

    # -- results -------------------------------------------------------------
    def live_trace(self):
        """(positions, values) for the points taken SO FAR -- callable from
        another thread while the scan is still running.

        The buffers are preallocated and each point is written before `n_taken`
        is bumped, so a reader can only ever see fully-written points. The
        returned slices are views: copy them if they must outlive the next point.
        """
        if self._buf_positions is None:
            return None, None
        n = self.n_taken
        return self._buf_positions[:n], self._buf_values[:n]

    def has_data(self) -> bool:
        return self.positions is not None and len(self.positions) > 0

    def as_dict(self) -> dict:
        """Arrays ready for `np.savez` (metadata goes in separately)."""
        out = {
            "positions_mm": self.positions,
            "targets_mm": self.targets,
            "values": self.values,
            "errors": self.errors,
            "reads": self.reads,
            "timestamps_s": self.timestamps,
        }
        if self.xyr is not None:
            out["x"] = self.xyr[0]
            out["y"] = self.xyr[1]
            out["r"] = self.xyr[2]
        return out


if __name__ == "__main__":
    from instruments.lockin_sr860 import LockInSR860
    from instruments.mcs2_stage import MCS2Stage

    st = MCS2Stage(); st.connect(simulate=True)
    li = LockInSR860(); li.connect("sim", "")
    sc = LockInScanner(st, li)
    pos, val = sc.scan(-0.05, 0.05, 11, parameter="R", dwell_s=0.05, status=print)
    for p, v, nr in zip(pos, val, sc.reads):
        print(f"{p:+.5f} mm  {v:+.6e}  ({nr} reads averaged)")
    li.disconnect(); st.disconnect()
