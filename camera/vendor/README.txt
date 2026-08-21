Put the Hamamatsu Orca device module here:

    camera\vendor\CameraDevice.py

That file must define the class `HamamatsuDevice` (the ScopeFoundry Orca Flash
DCAM device class by Castriotta/Zecchi/Bassi, Polimi). It talks to the camera
via ctypes -> dcamapi.dll directly, so nothing else needs installing beyond the
DCAM-API runtime that your Hamamatsu software already uses.

camera\hamamatsu_backend.py imports it from here automatically (see add_path()).
No folder outside the repo is required.

Then run:   python main.py --mode orca      (close the Hamamatsu proprietary
software first -- DCAM allows only one program to hold the camera at a time.)
