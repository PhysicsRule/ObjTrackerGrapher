# I have to have this as a seperate file bc putting it in camera.py gets a circular import
from src.tracker.lib.cameras.intel_realsense_D435i import IntelRealSenseD435i

camera = IntelRealSenseD435i()
