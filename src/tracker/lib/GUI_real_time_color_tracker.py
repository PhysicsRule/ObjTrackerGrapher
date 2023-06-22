## Real-time color tracker ##
# Tracks the largest object of a specified color and graphs the position on one graph with 3 lines representing the location in the x,y,z dimensions.
# The tracking is delayed a little bit due to aligning the color and depth frames as it tracks.

import os
import argparse
import queue
import threading
from typing import Callable, Tuple
import cv2
import numpy as np
import datetime
import pyrealsense2 as rs

from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtCore import *

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg          # In the future we should have the pyqtgraph locally stored. see discord redources
import time

from tracker.lib.setup_files import set_up_color, make_csv_files
from tracker.lib.user_input import make_new_folder
from tracker.lib.intel_realsense_D435i import get_all_frames_color, get_depth_meters, find_and_config_device, select_furthest_distance_color 
from tracker.lib.color import make_color_hsv, find_object_by_color_with_red, find_object_by_color
from tracker.lib.general import open_the_video, create_data_file, make_default_folder 
from tracker.lib.graphs_right_after import nine_graphs

def GUI_real_time_color_tracking(src, type_of_tracking, image ,color_ranges , min_radius_object, data_output_folder_path):

    app = pg.mkQApp("Realtime_Graphing")

    class DataGraph(pg.PlotWidget):
        def __init__(self):
            super().__init__()
            self.timestamps = np.array([], dtype=float)
            self.xpositions = np.array([], dtype=int)
            self.ypositions = np.array([], dtype=int)
            self.zpositions = np.array([], dtype=int)
            
            self.setBackground('w')
            styles = {'color':'r', 'font-size':'18px'}
            self.setLabel('bottom', 'Time (s)', **styles)
            self.setLabel('left', 'Position (m)', **styles)
            self.xpos = self.plot(pen=pg.mkPen('b',width=5))
            self.ypos = self.plot(pen=pg.mkPen('r',width=5))
            self.zpos = self.plot(pen=pg.mkPen('g',width=5))

            # Set the color to the first object in the array
            ## TODO maybe someday in real-time track 2 colors at once
            ## TODO How do I pass all of the information about the object? Do I call color_ranges in DataThread?
            self.color_ranges = color_ranges
            self.lower,self.upper,self.color,self.mass,self.radius_meters=color_ranges[0]
            self.image = image
            
            self._thread = DataThread(self.color, self.update_graph)
            # self._thread.run()

        def collect_data(self):
            self._thread.run()

        def update_graph(self, new_data: Tuple[float, float, float, float]):
            self.x_coord, self.y_coord, self.z_coord, timestamp = new_data
            # Append corresponding data points
            self.xpositions = np.append(self.xpositions, self.x_coord)
            self.ypositions = np.append(self.ypositions, self.y_coord)
            self.zpositions = np.append(self.zpositions, self.z_coord)

            self.timestamps = np.append(self.timestamps, timestamp)

            # Trim data if it goes too long
            t_range = 210                   # Number of data points to plot
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
                    print("didn't find pipeline frame")
                    continue
                            
                self.frame_queue.put(frame_result)

        def get_frame(self):
            if self.frame_queue.empty():
                return None
            return self.frame_queue.get()


    class DataThread(QtCore.QThread):
        new_data = QtCore.pyqtSignal(tuple)
        '''HSV_COLORMAP = {
            'green': ((29, 86, 6), (64, 255, 255)),
            'red': ((0,146,12), (11,255,206)),
            'frisbee': (( 0, 106,  87), (  7, 255, 255)),
            'double_red': ((0, 20, 6), (20, 255, 230), (160, 20, 6), (179, 255, 230)),
            'blue': ((95, 100, 80), (125, 255, 170)),
            'purple':((139,  68,  78), (170, 255, 255)),
            'yellow': ((20, 36, 4), (71, 238, 213)),
            'orange': ((0, 123, 189), (24, 255, 255)),
        }'''

        ## Setup the color tracker
        def __init__(self, color, callback):
            super().__init__()
            make_csv_files(color_ranges, data_output_folder_path)
            # Configure and setup the cameras
            self.pipeline = find_and_config_device()
            # OpenCV initialization

            # Find the furthest distance and in the future find a different origin TODO
            self.zeroed_x, self.zeroed_y, self.zeroed_z, self.z = select_furthest_distance_color(self.pipeline)
            self.camera_thread = CameraThread(self.pipeline)
            self.camera_thread.start()

            ## self.video_stream = VideoStream(src=0)
            self.start_time = None
            print('color in pipe', color)
            
            #self.current_colormap = self.HSV_COLORMAP[color]
            self.lower,self.upper, self.color, self.radius_meters, self.mass = color_ranges[0]           
            self.color = color
            self.color_ranges = color_ranges
            #self.current_colormap = self.HSV_COLORMAP[color]
            self.max_num_point=len(color_ranges)
            self.new_data.connect(callback)

        def run(self):
            image_file_path = os.path.abspath(os.path.join(data_output_folder_path + '/'+ self.color + '/'))  

            first_time_check = True
            start_time = 0 # It should get a time the first round through
            video_img_array = []
            # counter for the frames it saves
            i = 0

            # TODO: Between the time that this loop starts, and when the first frame is retrieved is
            # about ~1.2-1.6 seconds. Not sure the impact of that difference

            # Collect data   
            while True:

                # Get frames if valid
                frame_result = self.camera_thread.get_frame()
                if not frame_result:
                    continue
            
                (cv_color, rs_color, rs_depth), timestamp = frame_result
                            
                # Create a colormap from the depth data TODO add if desired
                #depth_image = np.asanyarray(rs_depth.get_data())
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)

                #from https://github.com/IntelRealSense/librealsense/issues/2204#issuecomment-414497056 

                hsv = make_color_hsv(cv_color) 
                x_pixel, y_pixel, radius, mask = find_object_by_color(cv_color,hsv, self.lower,self.upper, self.color, self.radius_meters, self.mass, min_radius_object, self.max_num_point)    
                if x_pixel is None:
                    continue
                # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)
                x_coord, y_coord, z_coord = get_depth_meters(x_pixel, y_pixel, self.radius_meters, rs_depth, rs_color, self.zeroed_x, self.zeroed_y, self.zeroed_z, self.z)
                if x_coord is None:
                    continue
                    
                # Start the timer
                if first_time_check:
                    start_time = timestamp
                    # we might want this later and compare with start that has milliseconds
                    first_time_check = False

                    # Converts time from milliseconds to seconds
                relative_timestamp = (timestamp - start_time) / 1000
                
                csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, self.color + '.csv'))   
                with open(csv_file_path, 'a') as data_to_file:
                    data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n')      

                cv2.putText(cv_color, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(cv_color, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(cv_color, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.putText(cv_color, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                cv2.circle(cv_color, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)
                # Create a colormap from the depth data
                if image.show_depth:
                    depth_image = np.asanyarray(rs_depth.get_data())
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)
                    cv2.circle(depth_colormap, (int(x_pixel), int(y_pixel)), int(radius), (255, 255, 255), 2)
                    # Show depth colormap & color feed
                    cv2.imshow('depth', depth_colormap)
                    cv2.moveWindow('depth',850,0)

                if image.show_RGB:
                    cv2.imshow('Tracking', cv_color)
                    cv2.moveWindow('Tracking',0,0)
                
                if image.show_mask and mask is not None:
                    cv2.imshow('mask', mask)
                    cv2.moveWindow('mask',0,500)
                

                # Save the RGB and depth images to view later if you want, but it does slow the tracking down a bit.
                if image.save_RGB:
                    color_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'color'+  str(i) + '.jpg'))   
                    cv2.imwrite(color_file_path,cv_color)
                if image.save_depth:
                    depth_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'depth'+  str(i) + '.jpg'))   
                    cv2.imwrite(depth_file_path, depth_colormap)
                if image.save_mask:
                    mask_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'mask'+  str(i) + '.jpg'))   
                    cv2.imwrite(mask_file_path, mask)
                if image.save_video:
                    height, width, layers = cv_color.shape
                    size = (width,height)
                    video_img_array.append(cv_color)
                i +=1
                
                self.new_data.emit((x_coord, y_coord, z_coord, relative_timestamp))
                
                # exit if spacebar or esc is pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 or k == 32:
                    print('end')
                    # Close all OpenCV windows
                    if image.save_video:
                        out = cv2.VideoWriter(image_file_path +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
                        for i in range(len(video_img_array)):
                            out.write(video_img_array[i])
                        out.release()
                    cv2.destroyAllWindows()
                    break


            
    # Make background white & foreground (axes) black
    pg.setConfigOptions(background=(255, 255, 255), foreground=(0, 0, 0))

    # Instantiate plot widget


    data_graph = DataGraph()
    data_graph.show()

    data_graph.collect_data()
    ## This is done to make sure the GUI starts up before data starts being collected
    ## Removed so we did not have to have a breakpoint in code
    #timer = QtCore.QTimer()
    #timer.timeout.connect(data_graph.collect_data)
    #timer.start(1000)

    if __name__ == "__main__":
        pg.mkQApp().exec_()   
        # nine_graphs()         

                    
