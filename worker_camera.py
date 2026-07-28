from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import time

import numpy as np
from loguru import logger

from camera.camera_interface import CameraStatus, MeasurementResult
from camera.factory import create_camera


def make_frame_metadata(frame: np.ndarray) -> MeasurementResult:
    peak = float(frame.max()) if frame.size else 0.0
    return MeasurementResult(
        peak=peak,
        total_power=0.0,
        centroid_x=0.0,
        centroid_y=0.0,
        beam_width_x=0.0,
        beam_width_y=0.0,
        timestamp=time.time(),
    )


def publish_status(frame_queue: mp.Queue, status: CameraStatus) -> None:
    # Status (incl. temperature) must get through even while streaming -- the
    # frame queue is usually full, so drop the oldest packet to make room.
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put({
        "type": "status",
        "status": {
            "connected": status.connected,
            "acquiring": status.acquiring,
            "backend": status.backend,
            "message": status.message,
            "width": status.width,
            "height": status.height,
            "serial_number": status.serial_number,
            "average_count": status.average_count,
            "exposure_ms": status.exposure_ms,
            "frame_counter": status.frame_counter,
            "raw_peak_count": status.raw_peak_count,
            "board_temp_c": status.board_temp_c,
            "fpa_temp_k": status.fpa_temp_k,
            "binning": status.binning,
            "roi_hsize": status.roi_hsize,
            "roi_vsize": status.roi_vsize,
            "roi_hpos": status.roi_hpos,
            "roi_vpos": status.roi_vpos,
            "sensor_width": status.sensor_width,
            "sensor_height": status.sensor_height,
        },
    })


def publish_frame(frame_queue, frame, measurement, shared_frame=None) -> None:
    # Vertical flip at the SINGLE source point: the sensor's row 0 images the
    # BOTTOM of the physical scene, so the raw frame is upside down. Flipping
    # here (before shared memory / the queue) makes every downstream consumer --
    # live view, ROI, background, the hyperspectral cubes, the analyzer, saved
    # TIFF/NPY -- physically correct, with no per-consumer convention to track.
    # A vertical flip only rearranges pixels spatially; per-pixel spectra are
    # unchanged. ascontiguousarray so the shared-memory write and the embedded
    # ndarray (pickled) path both get a plain C-contiguous array.
    frame = np.ascontiguousarray(frame[::-1])
    frame_height, frame_width = frame.shape
    if shared_frame is not None:
        if frame_height > shared_frame.shape[0] or frame_width > shared_frame.shape[1]:
            shared_frame = None
        else:
            shared_frame[:frame_height, :frame_width] = frame

    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

    packet = {
        "type": "frame",
        "measurement": {
            "peak": measurement.peak,
            "total_power": measurement.total_power,
            "centroid_x": measurement.centroid_x,
            "centroid_y": measurement.centroid_y,
            "beam_width_x": measurement.beam_width_x,
            "beam_width_y": measurement.beam_width_y,
            "timestamp": measurement.timestamp,
        },
        "shape": (frame_height, frame_width),
    }
    if shared_frame is None:
        packet["frame"] = frame
    else:
        packet["shared"] = True
    frame_queue.put(packet)


def camera_worker(frame_queue: mp.Queue, control_queue: mp.Queue, camera_config: dict) -> None:
    logger.info("Camera worker starting")
    mode = camera_config.get("mode", "irc806")
    target_fps = float(camera_config.get("target_fps", 30.0))
    frame_interval = 1.0 / max(target_fps, 1.0)
    camera = None
    shared_frame = None
    shared_memory_handle = None

    shared_frame_name = camera_config.get("shared_frame_name")
    shared_frame_shape = tuple(camera_config.get("shared_frame_shape", (0, 0)))
    if shared_frame_name and len(shared_frame_shape) == 2:
        shared_memory_handle = shared_memory.SharedMemory(name=shared_frame_name)
        shared_frame = np.ndarray(shared_frame_shape, dtype=np.uint16, buffer=shared_memory_handle.buf)

    def connect_camera(selected_mode: str) -> None:
        nonlocal camera, mode
        mode = selected_mode
        if camera is not None:
            camera.stop_acquisition()
            camera.disconnect()
        camera = create_camera(mode)
        status = camera.connect()
        status.requested_mode = mode
        publish_status(frame_queue, status)
        if status.connected:
            camera.start_acquisition()
            publish_status(frame_queue, camera.get_status())
        else:
            logger.warning(f"Camera unavailable: {status.message}")

    connect_camera(mode)

    running = True
    # Whether we WANT to be streaming (intent). Stays True across a link drop so
    # auto-reconnect keeps retrying even after status.acquiring flips to False.
    should_stream = camera is not None and camera.status.acquiring
    last_status_push = 0.0
    last_temp_push = 0.0
    last_frame_time = time.time()
    last_reconnect_try = 0.0
    RECONNECT_AFTER_S = 4.0       # no frames this long while "acquiring" = link lost
    RECONNECT_THROTTLE_S = 6.0    # don't hammer reconnect attempts
    try:
        while running:
            try:
                command = control_queue.get_nowait()
                cmd_type = command.get("type")
                if cmd_type == "stop":
                    running = False
                    continue
                if cmd_type == "set_exposure":
                    camera.set_exposure(float(command.get("value", 2.0)))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_average":
                    if hasattr(camera, "set_average"):
                        camera.set_average(int(command.get("value", 1)))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type in ("set_binning", "set_roi", "set_full_frame"):
                    # Binning / subarray can only change while idle, so briefly
                    # stop, apply, and restart if we were streaming.
                    was_streaming = camera.status.acquiring
                    camera.stop_acquisition()
                    if cmd_type == "set_binning" and hasattr(camera, "set_binning"):
                        camera.set_binning(int(command.get("value", 1)))
                    elif cmd_type == "set_roi" and hasattr(camera, "set_roi"):
                        camera.set_roi(int(command.get("hsize", 0)),
                                       int(command.get("vsize", 0)),
                                       int(command.get("hpos", 0)),
                                       int(command.get("vpos", 0)))
                    elif cmd_type == "set_full_frame" and hasattr(camera, "set_full_frame"):
                        camera.set_full_frame()
                    if was_streaming:
                        camera.start_acquisition()
                    last_frame_time = time.time()   # avoid a spurious reconnect
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "start":
                    camera.start_acquisition()
                    should_stream = True
                    last_frame_time = time.time()
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "pause":
                    camera.stop_acquisition()
                    should_stream = False
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "connect":
                    connect_camera(command.get("mode", mode))
                    should_stream = camera is not None and camera.status.acquiring
                    last_frame_time = time.time()
                    continue
                if cmd_type == "read_temp":
                    if hasattr(camera, "refresh_temperatures"):
                        try:
                            camera.refresh_temperatures()
                        except Exception:  # noqa: BLE001
                            pass
                    publish_status(frame_queue, camera.get_status())
                    last_temp_push = time.time()
                    continue
                if cmd_type == "load_state":
                    if hasattr(camera, "load_state"):
                        fn = str(command.get("value", ""))
                        camera.stop_acquisition()
                        ok = camera.load_state(fn)
                        if ok and hasattr(camera, "set_correction"):
                            camera.set_correction(True)   # ensure NUC'd output
                        camera.start_acquisition()
                        should_stream = camera.status.acquiring
                        last_frame_time = time.time()   # slow load -> avoid spurious reconnect
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_correction":
                    if hasattr(camera, "set_correction"):
                        camera.set_correction(bool(command.get("value")))
                    publish_status(frame_queue, camera.get_status())
                    continue
                if cmd_type == "set_bpr":
                    if hasattr(camera, "set_bpr"):
                        camera.set_bpr(bool(command.get("value")))
                    continue
                if cmd_type == "set_nuc_slot":
                    if hasattr(camera, "set_nuc_slot"):
                        camera.set_nuc_slot(int(command.get("value", 0)))
                    continue
                if cmd_type == "snapshot":
                    frame = camera.get_frame()
                    if frame is not None:
                        publish_frame(frame_queue, frame, make_frame_metadata(frame), shared_frame)
                    continue
            except queue.Empty:
                pass

            loop_start = time.time()
            frame = camera.get_frame()
            if frame is not None:
                last_frame_time = loop_start
                publish_frame(frame_queue, frame, make_frame_metadata(frame), shared_frame)
            else:
                now = time.time()
                if now - last_status_push > 1.0:
                    publish_status(frame_queue, camera.get_status())
                    last_status_push = now

                # Auto-reconnect: if we intend to stream but no frame has arrived
                # for a while, the GigE link dropped (NOT_CONNECTED). Keep trying
                # (throttled) until it returns -- self-healing across link drops.
                if (should_stream
                        and now - last_frame_time > RECONNECT_AFTER_S
                        and now - last_reconnect_try > RECONNECT_THROTTLE_S):
                    last_reconnect_try = now
                    if hasattr(camera, "reconnect"):
                        logger.warning("No frames -- attempting camera reconnect")
                        try:
                            camera.reconnect()
                        except Exception:  # noqa: BLE001
                            logger.exception("reconnect attempt failed")
                        publish_status(frame_queue, camera.get_status())
                        last_frame_time = time.time()   # grace before next try

            # Periodically read camera temperatures (serial-over-GigE) and push
            # status. Every 5 min (first read is immediate); failures ignored.
            now = time.time()
            if now - last_temp_push > 300.0:
                if hasattr(camera, "refresh_temperatures"):
                    try:
                        camera.refresh_temperatures()
                    except Exception:  # noqa: BLE001
                        pass
                    publish_status(frame_queue, camera.get_status())
                last_temp_push = now

            # Pace the loop. get_frame() already blocks at the camera's native
            # frame period (RetrieveBuffer), so only sleep for the *remaining*
            # slice of the target interval -- never add latency on top of the
            # hardware rate. With a high target_fps the camera sets the rate.
            if frame is None:
                time.sleep(0.005)
            else:
                remaining = frame_interval - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    except Exception:
        logger.exception("Camera worker crashed")
    finally:
        if camera is not None:
            camera.stop_acquisition()
            camera.disconnect()
            publish_status(frame_queue, camera.get_status())
        if shared_memory_handle is not None:
            shared_memory_handle.close()
        logger.info("Camera worker exited")
