from src.tracker.lib.cameras.camera import Camera
import depthai as dai
from typing import Any, Tuple, Optional


class OakDLite(Camera):

    def __init__(self):
        self.pipeline = None
        self.color_out: Optional[dai.MessageQueue] = None
        self.left_out: Optional[dai.MessageQueue] = None
        self.right_out: Optional[dai.MessageQueue] = None
        self.stereo_out: Optional[dai.MessageQueue] = None

    def warm_up_camera(self) -> None:
        pass

    def find_and_config_device(self):
        self.pipeline = dai.Pipeline()

        color = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        mono_left = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to color camera
        stereo.setOutputSize(640, 480)  # Set output size to match our requested resolution

        color_cam_out = color.requestOutput((640, 480))

        mono_left_out = mono_left.requestOutput((640, 480))
        mono_right_out = mono_right.requestOutput((640, 480))

        mono_left_out.link(stereo.left)
        mono_right_out.link(stereo.right)

        self.color_out = color_cam_out.createOutputQueue()
        self.right_out = mono_right_out.createOutputQueue()
        self.left_out = mono_left_out.createOutputQueue()
        self.stereo_out = stereo.depth.createOutputQueue()

        self.pipeline.start()

    def find_and_config_device_mult_stream(self, types_of_streams_saved):
        self.find_and_config_device()

    def record_bag_file(self, data_output_folder_path, types_of_streams_saved):
        pass

    def get_all_frames_color(self) -> Optional[Tuple[Tuple[Any, Any, Any], Any]]:
        color_frame = self.color_out.get()
        stereo_frame = self.stereo_out.get()

        assert color_frame.validateTransformations()
        assert stereo_frame.validateTransformations()

        color = color_frame.getCvFrame()
        depth = stereo_frame.getDepthFrame()


    def get_all_frames_infrared(self) -> Optional[Tuple[Tuple[Any, Any, Any], Any]]:
        pass

    def select_clipping_distance(self, frame, depth_frame) -> Tuple[Any, float, float]:
        pass

    def get_coordinates_meters(self, frame, depth_frame, x_pixel: int, y_pixel: int, depth) -> Tuple[float, float, float]:
        pass

    def select_location(self, frame, color_frame, depth_frame) -> Tuple[float, Tuple[float, float, float], float]:
        pass

    def get_depth_meters(self, x_pixel, y_pixel, radius_meters, depth_frame, color_frame, zeroed_x, zeroed_y, zeroed_z, clipping_distance) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        pass

    def set_the_origin(self, point) -> Tuple[float, float, float]:
        pass

    def select_furthest_distance_color(self) -> Tuple[float, float, float, float]:
        pass

    def select_furthest_distance_infrared(self):
        pass

    def read_bag_file_and_config(self, types_of_streams_saved, data_output_folder_path, folder_name, bag_folder_path) -> Any:
        pass