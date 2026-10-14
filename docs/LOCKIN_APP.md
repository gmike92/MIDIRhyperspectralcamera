# Lock-in acquisition app (MCS2 + SR865A)

`main_lockin.py` is the imaging acquisition app with the camera replaced by a
lock-in amplifier. It step-scans a **SmarAct MCS2** translation stage and reads
a **Stanford Research SR865A** at every commanded position, plotting the
intensity trace against stage position while the scan runs.

```
python main_lockin.py                # real hardware
python main_lockin.py --simulate     # fake stage + fake signal, no hardware
run_lockin.bat                       # same, via the repo's .venv
```

The camera app (`main.py`) is untouched and still runs exactly as before; the
two share `instruments/` but nothing else.

## What replaced what

| Camera app                              | Lock-in app                            |
| --------------------------------------- | -------------------------------------- |
| Camera panel (`_build_camera_group`)    | **Lock-in panel** — `ui/lockin_panel.py` |
| Live image + X/Y profiles               | **Scan trace** (value vs position) + **live monitor** (value vs time) |
| ROI mean per frame                      | Lock-in channel read per position      |
| `worker_camera.py` subprocess           | none — both instruments are driven from the GUI process |
| TWINS wedge (SCU3D)                     | MCS2 stage — `ui/mcs2_panel.py`        |

## Layout

**Left — instrument tabs**

- **Lock-in** — interface + address, connect, time constant, filter slope,
  sensitivity, phase, auto phase/range/scale, and the **channel** the scan
  records.
- **Stage** — MCS2 locator + channel, connect, find reference, calibrate,
  absolute move, jog, velocity/acceleration, stop.
- **Scan** — start/stop/steps, the dwell time, and saving.

**Right — the live view**

- **Scan trace** — the recorded channel against stage position **in µm**,
  growing point by point as the scan runs. An orange marker sits on the point
  being measured, and *Keep previous scan* leaves the previous trace behind it
  as a grey dashed ghost so a change in alignment is obvious.

  > µm is display-only: positions travel in **mm** everywhere else — the stage
  > readout and Go-to box, the Scan tab's Start/Stop, and the saved
  > `positions_mm` / `targets_mm` columns. The conversion happens once, in
  > `ui/lockin_window.py` (`UM_PER_MM`), on the way onto the plot. A typical
  > scan spans tens of µm, which in mm is an axis full of `0.0500`-style labels.
- **Live monitor** — the same channel against time, at 5 Hz, whenever the
  lock-in is connected and no scan is running. This is the strip chart that
  stands in for the camera's live view when lining the experiment up by hand.
- Big readouts of the current value and stage position, legible from across the
  bench.

## How the two instruments are synchronised

Per point, in `instruments/lockin_scan.py`:

1. **Move** — `MCS2Stage.move_to(target)` (closed-loop absolute).
2. **Handshake** — `wait_for_stop()` polls `CHANNEL_STATE` until
   `ACTIVELY_MOVING` clears, and fails loudly on `MOVEMENT_FAILED` (end stop,
   range limit, following error). MCS2 move commands are asynchronous, so this
   handshake is what makes the scan a *step* scan rather than a race.
3. **Dwell** — read the lock-in as fast as the link allows for **dwell time**
   seconds. The point is the **mean of every reading taken** in that window,
   its error bar is their standard deviation, and `reads` records how many
   there were (typically thousands per second over LAN/USB).

**Dwell time is the only timing control.** Longer dwell = better averaging and
a slower scan; the *Estimate* line turns it into a total scan duration before
you start.

### Keep the dwell longer than the time constant

The window average includes the lock-in's settling transient at the start — its
output is the input through a low-pass filter, so it takes a few time constants
to forget the previous position. That bias is negligible while the dwell is
comfortably longer than τ and grows as the two approach each other.

The panel reads τ off the instrument and shows the ratio under the dwell box:

```
5 × the 0.1 s time constant                       <- fine
dwell 0.05 s is only 0.5 x the 0.1 s time constant
  -- the average is biased by the settling transient.
  Use at least 0.5 s, or shorten the time constant on the lock-in.
```

Below **5 τ** it turns red and the warning is saved with the scan as
`dwell_warning`. Either lengthen the dwell or shorten τ on the lock-in — both
fix it.

> **Symptom of too short a dwell:** reduced amplitude, washed-out fringe
> contrast, and a trace shifted along the position axis. The definitive test is
> to scan the range forward, then swap Start and Stop and scan back — with
> enough dwell the two traces overlay; if they are shifted apart, the dwell is
> still too short.

### The stage is checked for actually arriving

MCS2 movement commands are documented *not* to report an error when the move
cannot be processed — a wrong channel index, an unplugged positioner or a range
limit makes `Move` silently do nothing, `wait_for_stop()` return at once, and
the position never change. A scan would then finish looking perfectly normal
with every point carrying the wrong position.

So each point asserts the read-back position is within tolerance of the target
(the larger of 2 µm and 20 % of the step, recorded as `position_tolerance_mm`),
and stops the scan naming what to check if not. The scan also warns up front if
the stage reports no position sensor or has never been referenced.

Optionally the quadratures are averaged over the same window too, so the scan
can be re-phased afterwards (*Also record X, Y, R*). That is free when the
recorded channel is X, Y or R — a single `SNAP?` yields the point value and all
three quadratures from the same instant — and halves the read rate otherwise.

While a scan runs, both instrument panels are frozen — their poll timers stop
and their controls grey out — so the scan thread is the only caller talking to
either device.

## Choosing the channel

The SR865A has two different things called "channels", and both are in the
**Channel** dropdown:

- **Output parameters** (`OUTP?`) — `X`, `Y`, `R`, `Theta`, `Aux In 1-4`,
  `X noise`, … Always available, whatever the front panel is showing.
- **Display data channels** (`OUTR?`) — `Data 1`-`Data 4`, i.e. whatever the
  four front-panel display slots are currently assigned to.

Pick `R` for a plain intensity trace, or a `Data n` slot to record exactly what
the front panel is set up to show.

## Connecting

**Lock-in.** Four interfaces, chosen in the panel:

| Interface        | Address                                      | Needs                       |
| ---------------- | -------------------------------------------- | --------------------------- |
| LAN (VXI-11)     | `192.168.1.10`                               | `python-vxi11` *(default)*  |
| VISA (USB/GPIB)  | `USB0::0xB506::0x2000::00xxxxx::INSTR`       | `pyvisa` **+ a VISA runtime** |
| LAN (raw TCP)    | `192.168.1.10:23`                            | nothing                     |
| Simulate         | —                                            | nothing                     |

*Find VISA* lists what is attached and picks the SRS device if it sees one.
Everything the app needs is plain SCPI text, so the driver is transport-agnostic.
`srsinst.sr860` (installed from `requirements.txt`) is used when importable; the
git clone at `instruments/srsinst.sr860` is only a fallback for a machine where
that package was never installed, and never shadows it. Failing both, the driver
talks to `vxi11`/`pyvisa` directly. `LockInSR860.backend` records which was used.

### USB needs more than `pip install pyvisa`

USB over VISA needs **two** things, and `pip install pyvisa` gives neither:

1. a **VISA implementation** — otherwise `pyvisa.ResourceManager()` raises
   *"Could not locate a VISA implementation"*;
2. a **Windows driver bound to the instrument** — otherwise the SR865A sits in
   Device Manager with an error and no driver class, and nothing can reach it.

| Route | VISA impl. | USB driver | Verdict |
| ----- | ---------- | ---------- | ------- |
| **NI-VISA** (ni.com) or **Keysight IO Libraries** | ✅ | ✅ USBTMC | recommended — one installer, both halves |
| `pip install pyvisa-py pyusb libusb-package` | ✅ (auto-selected) | ❌ — needs **Zadig** to rebind the device to WinUSB | works, but fiddly: Zadig rebinding persists, breaks vendor software expecting USBTMC, and hides the device from NI-VISA if you install it later |
| **LAN (VXI-11)** | not needed | not needed | simplest if the instrument is on the network |

Once a driver is bound, the device enumerates and its resource string is
`USB0::0xB506::0x2000::<serial>::INSTR`. No app changes are needed to use
pyvisa-py: plain `ResourceManager()` picks it up automatically when it is the
only backend installed.

*Find VISA* distinguishes the three failure modes — pyvisa missing, no runtime
behind it, runtime working but nothing attached — and says which applies.

> Installing pyvisa *without* a runtime also breaks `import srsinst.sr860`: its
> `VisaInterface` builds a `ResourceManager` at class-body scope, so the import
> itself raises. The app degrades gracefully (it then talks to `vxi11`/`pyvisa`
> directly and `backend` reads `vxi11` instead of `srsinst-vxi11`), but it is a
> confusing side effect if you are wondering why the SRS driver stopped loading.

If you only need LAN, skip all of this: **VXI-11 needs no runtime.**

**Stage.** *Scan* lists the MCS2 controllers reachable from this PC and fills in
the locator (`usb:sn:MCS2-…` or `network:…`); set the channel the positioner is
on and connect.

> The MCS2 position sensors are **incremental**. Absolute positions only mean
> something after **Find reference** has run once per power-up. *Calibrate* is a
> separate, rarer thing — it moves the positioner by up to several millimetres
> and is only needed when the mechanics or the positioner type changed.

Between points the stage holds position indefinitely (`HOLD_TIME_INFINITE`),
because a step scan can dwell for seconds and the default finite hold would let
the positioner drift mid-reading. `disconnect()` and **Stop** release the hold.

## Saved files

Each scan writes `<YYYYmmdd_HHMMSS>.<filename>.{csv,npz,json}` into the Save
folder (automatically when the scan finishes, if *Save automatically* is on).
*Export plot* adds a `.png` of the trace.

| Column / key   | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| `positions_mm` | position **read back from the encoder** at each point   |
| `targets_mm`   | position **commanded**                                  |
| `values`       | the recorded channel                                    |
| `errors`       | standard deviation of the readings averaged at that point |
| `reads`        | how many readings went into that point                   |
| `timestamps_s` | seconds since the scan started                          |
| `x`, `y`, `r`  | quadrature snapshot, if *Also record X, Y, R* was on    |

The `.json` (also embedded in the `.npz` as `metadata`) holds the full scan
configuration and the lock-in's settings at scan time: `dwell_s`,
`dwell_per_tau`, any `dwell_warning`, `reads_per_point_mean`, τ, filter slope,
sensitivity, phase, reference frequency, and the stage identification. An aborted scan saves the points it did take, truncated.

## Files

```
main_lockin.py                 entry point
run_lockin.bat                 launcher
instruments/mcs2_stage.py      SmarAct MCS2 driver (mm interface, + simulate)
instruments/lockin_sr860.py    SR865A driver (VXI-11 / VISA / TCP / simulate)
instruments/lockin_scan.py     the step-scan engine (no GUI)
ui/lockin_panel.py             lock-in panel
ui/mcs2_panel.py               stage panel
ui/lockin_scan_panel.py        scan setup, run, save
ui/lockin_window.py            main window + plots
ui/device_controller.py        shared "run blocking driver calls off the GUI thread"
```

Every driver has a `simulate` mode and a `__main__` self-test, so the scan logic
can be exercised with nothing plugged in:

```
python -m instruments.lockin_scan     # simulated stage + simulated interferogram
```
