from thorcam.camera import ThorCam
import numpy as np
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

NUM_FRAMES = 10  # adjust to the desired number of frames

from instruments.twins_stage import TwinsStage

st = TwinsStage()
if not st.connect(home=False, reference=False):
    raise SystemExit("Smaract stage NOT connected")
print("Smaract stage connected")
referenced = st.is_referenced()
print("referenced:", referenced)
print('Initial position:', st.get_position())


with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("no cameras detected")

    with sdk.open_camera(available_cameras[0]) as camera:
        camera.exposure_time_us = 11000  # set exposure to 11 ms
        camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        camera.image_poll_timeout_ms = 1000  # 1 second polling timeout
        old_roi = camera.roi  # store the current roi
        """
        uncomment the line below to set a region of interest (ROI) on the camera
        """
        # camera.roi = (100, 100, 600, 600)  # set roi to be at origin point (100, 100) with a width & height of 500

        """
        uncomment the lines below to set the gain of the camera and read it back in decibels
        """
        #if camera.gain_range.max > 0:
        #    db_gain = 6.0
        #    gain_index = camera.convert_decibels_to_gain(db_gain)
        #    camera.gain = gain_index
        #    print(f"Set camera gain to {camera.convert_gain_to_decibels(camera.gain)}")

        camera.arm(2)

        camera.issue_software_trigger()

        # for i in range(NUM_FRAMES):
        #     frame = camera.get_pending_frame_or_null()
        #     if frame is not None:
        #         print("frame #{} received!".format(frame.frame_count))

        #         frame.image_buffer  # .../ perform operations using the data from image_buffer

        #         #  NOTE: frame.image_buffer is a temporary memory buffer that may be overwritten during the next call
        #         #        to get_pending_frame_or_null. The following line makes a deep copy of the image data:
        #         image_buffer_copy = np.copy(frame.image_buffer)
        #     else:
        #         print("timeout reached during polling, program exiting...")
        #         break

        # Scan
        pos = np.linspace(2, 2.5, 400)
        real_pos = []
        HyperMatrix = []
        for p in pos:
            st.move_to(p)
            st.wait_for_stop()
            # print("pos:", st.get_position(), "mm")        
            frame = camera.get_pending_frame_or_null()
                #     if frame is not None:
                #         print("frame #{} received!".format(frame.frame_count))
            HyperMatrix.append(np.copy(frame.image_buffer))
            real_pos.append(st.get_position())

        # Saving
        from scipy.io import savemat
        from datetime import datetime
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = rf'C:\Users\marti\Desktop\Lab_temp\CUNY\YVO\{timestamp}_alignment.mat'
        savemat(filename, {
            'raw_interferograms': HyperMatrix,
            'raw_positions': real_pos
        })

        # Disconnect
        camera.disarm()
        camera.roi = old_roi  # reset the roi back to the original roi
        st.disconnect()
#  Because we are using the 'with' statement context-manager, disposal has been taken care of.

print("program completed")




    