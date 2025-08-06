from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

class Camera(ABC):

    @abstractmethod
    def warm_up_camera(self) -> None:
        pass

    @abstractmethod
    def find_and_config_device(self):
        pass

    @abstractmethod
    def find_and_config_device_mult_stream(self, types_of_streams_saved):
        pass

    @abstractmethod
    def record_bag_file(self, data_output_folder_path, types_of_streams_saved):
        pass

    @abstractmethod
    def get_all_frames_color(self) -> Optional[Tuple[Tuple[Any, Any, Any], Any]]:
        pass

    @abstractmethod
    def get_all_frames_infrared(self) -> Optional[Tuple[Tuple[Any, Any, Any], Any]]:
        pass

    @abstractmethod
    def select_clipping_distance(self, frame, depth_frame) -> Tuple[Any, float, float]:
        pass

    @abstractmethod
    def get_coordinates_meters(self, frame, depth_frame, x_pixel: int, y_pixel: int, depth) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def select_location(self, frame, color_frame, depth_frame) -> Tuple[float, Tuple[float, float, float], float]:
        pass

    @abstractmethod
    def get_depth_meters(self, x_pixel, y_pixel, radius_meters, depth_frame, color_frame, zeroed_x, zeroed_y, zeroed_z, clipping_distance) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        pass

    @abstractmethod
    def set_the_origin(self, point) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def select_furthest_distance_color(self) -> Tuple[float, float, float, float]:
        pass

    @abstractmethod
    def select_furthest_distance_infrared(self):
        pass

    @abstractmethod
    def read_bag_file_and_config(self, types_of_streams_saved, data_output_folder_path, folder_name, bag_folder_path) -> Any:
        pass