
## Real-time color tracker ##

import os
import argparse
import queue
import threading
import cv2
import numpy as np
import datetime
import pyrealsense2 as rs

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg          # In the future we should have the pyqtgraph locally stored. see discord redources
from pyqtgraph.ptime import time



from tracker.lib.user_input import make_new_folder
from tracker.lib.intel_realsense_D435i import get_all_frames_color, get_depth_meters, find_and_config_device, select_furthest_distance_color 
from tracker.lib.color import make_color_hsv, find_object_by_color_with_red
from tracker.lib.general import open_the_video, create_data_file, make_default_folder 


# TODO use argparse values in config file later
ap = argparse.ArgumentParser()

ap.add_argument("-s", "--src", default =0,
    help="video camera the port uses")
ap.add_argument("-f", "--save_image", default =False,
    help="Saves each image captured to the folder")
ap.add_argument("-i", "--show_image", default =False,
    help="Shows each image captured on the screen")


args = vars(ap.parse_args())
src = args["src"]
# SET to True if you want all of the infrared images saved
save_image = args["save_image"] # image
# SET to True if you want to show the ball getting tracked. Note: This will reduce your framerate
show_image = args["show_image"]
# TODO Maybe add this later:
#ap.add_argument("-v", "--video",
#    help="path to the (optional) video file")

app = pg.mkQApp("Realtime Graphing Test")

class DataGraph(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.timestamps = np.array([], dtype=float)
        self.xpositions = np.array([], dtype=int)
        self.ypositions = np.array([], dtype=int)
        self.zpositions = np.array([], dtype=int)
          
        self.xpos = self.plot(pen=pg.mkPen('b',width=5))
        self.ypos = self.plot(pen=pg.mkPen('r',width=5))
        self.zpos = self.plot(pen=pg.mkPen('g',width=5))

        self.color = input("What color is your object (orange, green, red, yellow, blue, purple)? ")
        self._thread = DataThread(self.color, self.update_graph)
        # self._thread.run()

    def collect_data(self):
        self._thread.run()

    def update_graph(self, new_data):
        self.x_coord, self.y_coord, self.z_coord, timestamp = new_data

        # Append corresponding data points
        self.xpositions = np.append(self.xpositions, self.x_coord)
        self.ypositions = np.append(self.ypositions, self.y_coord)
        self.zpositions = np.append(self.zpositions, self.z_coord)

        self.timestamps = np.append(self.timestamps, timestamp)

        # Trim data if it goes too long
        t_range = 240                   # Number of data points to plot
        if self.timestamps.size > t_range:
            self.timestamps = self.timestamps[-t_range:]
            self.xpositions = self.xpositions[-t_range:]
            self.ypositions = self.ypositions[-t_range:]
            self.zpositions = self.zpositions[-t_range:]
        # Update pyqtgraph plots
        self.xpos.setData(x=self.timestamps, y=self.xpositions)
        self.ypos.setData(x=self.timestamps, y=self.ypositions)
        self.zpos.setData(x=self.timestamps, y=self.zpositions)

        app.processEvents()

class CameraThread(threading.Thread):
    def __init__(self,pipeline):
        super(CameraThread, self).__init__()

        
        self.pipeline=pipeline
        self.frame_queue=queue.Queue()

    def run(self):
        self.start_time = datetime.datetime.now()
        while True:
            frame_result = get_all_frames_color(self.pipeline)
            if not frame_result:
                continue
                        
            self.frame_queue.put(frame_result)

    def get_frame(self):
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()


class DataThread(QtCore.QThread):
    new_data = QtCore.pyqtSignal(tuple)
    HSV_COLORMAP = {
        'green': ((29, 86, 6), (64, 255, 255)),
        'red': ((0,146,12), (11,255,206)),
        'frisbee': (( 0, 106,  87), (  7, 255, 255)),
        'double_red': ((0, 20, 6), (20, 255, 230), (160, 20, 6), (179, 255, 230)),
        'blue': ((95, 100, 80), (125, 255, 170)),
        'purple':((139,  68,  78), (170, 255, 255)),
        'yellow': ((20, 36, 4), (71, 238, 213)),
        'orange': ((0, 123, 189), (24, 255, 255)),
    }

    ## Setup the color tracker
    def __init__(self, color, callback):
        super().__init__()
        
        # Configure and setup the cameras
        self.pipeline = find_and_config_device()
        # OpenCV initialization
         
        # Find the furthest distance and in the future find a different origin TODO
        self.zeroed_x, self.zeroed_y, self.zeroed_z, self.z = select_furthest_distance_color(self.pipeline)
        
        self.camera_thread = CameraThread(self.pipeline)
        self.camera_thread.start()

        ## self.video_stream = VideoStream(src=0)
        self.start_time = None
        self.color = color
        self.current_colormap = self.HSV_COLORMAP[color]

        self.new_data.connect(callback)

    def run(self):
        data_input = 'color_i'
        data_output = 'color_o'
        data_output_folder, data_output_folder_path = make_default_folder(data_input, data_output, self.color)
        csv_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ self.color + '.csv'))  
        create_data_file(csv_file_path)

        
        image_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ self.color + '/'))  

        first_time_check = True
        start_time = 0 # It should get a time the first round through
                # SET to True if you want all of the infrared images saved
        save_image = False # image
        # SET to True if you want to show the ball getting tracked. Note: This will reduce your framerate
        show_image = True
        show_depth = False
        # counter for the frames it saves
        i = 0
        radius_meters = 0.0 ## TODO change later to get color from code Assume it is a flat surface for now

        # Collect data   
        while True:
            # Get frames if valid
            frame_result = self.camera_thread.get_frame();
            if not frame_result:
                continue
        
            (cv_color, rs_color, rs_depth), timestamp = frame_result
                        
            # Create a colormap from the depth data TODO add if desired
            #depth_image = np.asanyarray(rs_depth.get_data())
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)

            #from https://github.com/IntelRealSense/librealsense/issues/2204#issuecomment-414497056

            x_pixel, y_pixel, radius, mask = find_object_by_color_with_red(cv_color, self.color, self.current_colormap)     

            if x_pixel is None:
                continue
            # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)
            x_coord, y_coord, z_coord = get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_color, self.zeroed_x, self.zeroed_y, self.zeroed_z, self.z)
            if x_coord is None:
                continue
                
            # Start the timer
            if first_time_check:
                start_time = timestamp
                # we might want this later and compare with start that has milliseconds
                first_time_check = False

                # Converts time from milliseconds to seconds
            relative_timestamp = (timestamp - start_time) / 1000

            # Writes the coordinates to each colored object
            #csv_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ color_name + '.csv'))   
            with open(csv_file_path, 'a') as data_to_file:
                data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n')      

            cv2.putText(cv_color, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv2.putText(cv_color, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv2.putText(cv_color, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv2.putText(cv_color, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            cv2.circle(cv_color, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)
            # Create a colormap from the depth data
            if show_depth:
                depth_image = np.asanyarray(rs_depth.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)
                cv2.circle(depth_colormap, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)
                # Show depth colormap & color feed
                cv2.imshow('depth', depth_colormap)
                cv2.moveWindow('depth',700,0)

            cv2.imshow('Tracking', cv_color)
            cv2.moveWindow('Tracking',0,0)
            i +=1

            if save_image:
                cv2.imwrite(image_file_path + str(i) +'.jpg', cv_color)
                cv2.imwrite(image_file_path + str(i) +'.jpg', depth_colormap)
            # Show masks
            # ## cv2.imshow('mask', mask)
            ## cv2.moveWindow('mask',1000,0)
            
            self.new_data.emit((x_coord, y_coord, z_coord, relative_timestamp))

            # exit if spacebar or esc is pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 or k == 32:
                print('end')
                # Cleanup after loop break

                # Stop OpenCV/RealSense video
                #cv_video.release()

                #pipeline.stop()
                # Close all OpenCV windows
                cv2.destroyAllWindows()
                break
        
# Make background white & foreground (axes) black
pg.setConfigOptions(background=(255, 255, 255), foreground=(0, 0, 0))

# Instantiate plot widget


data_graph = DataGraph()
data_graph.show()

# This is done to make sure the GUI starts up before data starts being collected
timer = QtCore.QTimer()
timer.timeout.connect(data_graph.collect_data)
timer.start(1000)

if __name__ == "__main__":
    pg.mkQApp().exec_()   
    # nine_graphs()         

                
