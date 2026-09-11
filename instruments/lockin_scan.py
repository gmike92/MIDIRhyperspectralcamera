"""
lockin_scan.py -- step-scan engine: SmarAct MCS2 stage + SR865A lock-in.

Sweeps the MCS2 stage across [start, stop] in N steps and, at each commanded
position, reads the demodulated signal from the lock-in. This is the piece that
"synchronises" the two instruments; everything about that synchronisation lives
in `_dwell_and_read`:

    move -> wait for the channel to stop moving   (hardware handshake)
         -> settle for  settle_factor x time_constant + extra_settle
         -> average `samples` reads, `sample_interval` apart

The settle is the part people get wrong. A lock-in's output is the input
convolved with its low-pass filter, so after any step change the reading needs
several time constants to forget the previous position. The rule of thumb built
in here is 5 tau for a 24 dB/oct filter (~99.9% settled); a 6 dB/oct filter is
happy with ~3 tau. `settle_factor` is exposed so it can be tightened for speed
or loosened for accuracy, and `recommended_settle_factor()` picks a default from
the filter slope actually configured on the instrument.

Hardware-only (NO GUI): progress and abort are plain callbacks, so the same
scanner drives the Qt panel and a headless script.

Usage:
    from instruments.lockin_scan import LockInScanner
    sc = LockInScanner(stage, lockin)
    pos, val = sc.scan(-0.1, 0.1, 201, parameter="R", samples=4)
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

#: Settle time as a multiple of the lock-in time constant, per filter slope.
#: A steeper filter rings longer, so it needs more taus to reach the same
#: fractional settling error.
SETTLE_FACTOR_BY_SLOPE = {6: 3.0, 12: 4.0, 18: 5.0, 24: 6.0}

DEFAULT_SETTLE_FACTOR = 5.0
DEFAULT_EXTRA_SETTLE_S = 0.0
DEFAULT_SAMPLES = 1


def recommended_settle_factor(filter_slope_db_oct: int) -> float:
    """Taus to wait after a step, for the given low-pass roll-off."""
    return SETTLE_FACTOR_BY_SLOPE.get(int(filter_slope_db_oct), DEFAULT_SETTLE_FACTOR)


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
        self.values = None          # the selected lock-in parameter
        self.errors = None          # std dev across the samples averaged per point
        self.xyr = None             # (3, N) X, Y, R when record_xyr was on, else None
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
    def _dwell_and_read(self, parameter: str, settle_s: float, samples: int,
                        sample_interval_s: float, record_xyr: bool):
        """Settle at the current position, then read. Returns (mean, std, xyr)."""
        if settle_s > 0:
            time.sleep(settle_s)
        mean, std = self.lockin.read_average(parameter, samples, sample_interval_s)
        xyr = None
        if record_xyr:
            try:
                xyr = self.lockin.snap("X", "Y", "R")
            except Exception:  # noqa: BLE001
                xyr = [float("nan")] * 3
        return mean, std, xyr

    # -- the scan ------------------------------------------------------------
    def scan(self, start_mm: float, stop_mm: float, n_steps: int,
             parameter: str = "R",
             samples: int = DEFAULT_SAMPLES,
             sample_interval_s: Optional[float] = None,
             settle_factor: Optional[float] = None,
             extra_settle_s: float = DEFAULT_EXTRA_SETTLE_S,
             record_xyr: bool = True,
             move_timeout_s: float = 30.0,
             progress: Optional[Callable[[int, int, float, float], None]] = None,
             should_abort: Optional[Callable[[], bool]] = None,
             status: Optional[Callable[[str], None]] = None):
        """Sweep [start_mm, stop_mm] in `n_steps` and read `parameter` at each.

        Step size = abs(stop-start)/(n_steps-1).

        `sample_interval_s` defaults to one time constant, so averaged samples
        are (nearly) independent rather than N copies of the same filtered value.
        `settle_factor` defaults to the value recommended for the instrument's
        current filter slope. Returns (positions_mm, values), both truncated to
        the points actually taken if the scan was aborted.
        """
        lockin, stage = self.lockin, self.stage
        n = int(n_steps)
        targets = np.linspace(float(start_mm), float(stop_mm), n)

        # Read the lock-in's own settings once and derive the timing from them,
        # so the scan adapts to whatever the instrument is actually set to.
        try:
            tau = float(lockin.time_constant)
        except Exception:  # noqa: BLE001
            tau = 0.1
        try:
            slope = int(lockin.filter_slope)
        except Exception:  # noqa: BLE001
            slope = 24
        if settle_factor is None:
            settle_factor = recommended_settle_factor(slope)
        if sample_interval_s is None:
            sample_interval_s = tau
        settle_s = settle_factor * tau + float(extra_settle_s)

        self.parameter = parameter
        self.metadata = {
            "parameter": parameter,
            "start_mm": float(start_mm),
            "stop_mm": float(stop_mm),
            "n_steps": n,
            "step_um": (abs(float(stop_mm) - float(start_mm)) / (n - 1) * 1000.0
                        if n > 1 else 0.0),
            "samples_per_point": int(max(1, samples)),
            "sample_interval_s": float(sample_interval_s),
            "settle_factor_tau": float(settle_factor),
            "extra_settle_s": float(extra_settle_s),
            "settle_s": float(settle_s),
            "time_constant_s": tau,
            "filter_slope_dB_oct": slope,
        }
        self.metadata.update(
            {f"lockin_{k}": v for k, v in lockin.settings_snapshot().items()})

        if status:
            status(f"tau={tau:g} s, settle={settle_s:.3f} s, "
                   f"{max(1, samples)} sample(s)/point")

        positions = np.full(n, np.nan)
        values = np.full(n, np.nan)
        errors = np.full(n, np.nan)
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
            # Keep the simulator's fake signal tied to the real scan geometry.
            if getattr(lockin, "backend", None) == "sim":
                lockin.set_sim_position(positions[i])

            try:
                mean, std, snap = self._dwell_and_read(
                    parameter, settle_s, samples, sample_interval_s, record_xyr)
            except Exception as e:  # noqa: BLE001
                if status:
                    status(f"lock-in read failed at {target:.5f} mm: {e}")
                break

            values[i] = mean
            errors[i] = std
            if xyr is not None and snap is not None:
                xyr[:, i] = snap
            stamps[i] = time.time() - t0
            taken = i + 1
            self.n_taken = taken        # after the writes: never publish a hole
            if progress:
                progress(taken, n, float(positions[i]), float(values[i]))

        self.targets = targets[:taken]
        self.positions = positions[:taken]
        self.values = values[:taken]
        self.errors = errors[:taken]
        self.xyr = xyr[:, :taken] if xyr is not None else None
        self.timestamps = stamps[:taken]
        self.metadata["points_taken"] = int(taken)
        self.metadata["duration_s"] = float(time.time() - t0)
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
    pos, val = sc.scan(-0.05, 0.05, 21, parameter="R", samples=2,
                       settle_factor=0.0, sample_interval_s=0.0,
                       status=print)
    for p, v in zip(pos, val):
        print(f"{p:+.5f} mm  {v:+.6e}")
    li.disconnect(); st.disconnect()
