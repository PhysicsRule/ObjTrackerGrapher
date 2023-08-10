## Color Tracker

import os
import argparse
import cv2
import numpy as np

from tracker.lib.setup_files import set_up_color, make_csv_files
from tracker.lib.intel_realsense_D435i import get_all_frames_color, get_depth_meters, find_and_config_device, select_furthest_distance_color 
from tracker.lib.color import make_color_hsv, find_object_by_color
from tracker.lib.general import open_the_video 
from tracker.lib.user_input import make_new_folder

## Main Program ##
# TODO use argparse values in config file later
#ap = argparse.ArgumentParser()

#ap.add_argument("-s", "--src", default =0,
#    help="video camera the port uses")

    
#args = vars(ap.parse_args())
#src = args["src"]

# List cameras plugged into the machine so user can change the src value for their computer
    # checks the first 10 indexes.
index = 0
arr = []
src_count = 6
while index<src_count > 0:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.read()[0]:
        print(index, " ","'{}'".format(cv2.VideoCapture(index, cv2.CAP_DSHOW)))
        cap.release()
    index += 1

# user can change the src
src = int(input ('Which camera would you like?'))

# TODO Maybe add this later:
#ap.add_argument("-v", "--video",
#    help="path to the (optional) video file")

## GUI TODO Add note to reader that if they want to save the images, 
## the data rate is 20 fps and if they don't it is 30 fps.
## Showing the image does not affect the data rate very much
## Show a warning on the screen if they choose all show_images to be false as they will not be sure they are collecting data
## These are the defaults for the user when they first install the program
show_image = True
save_image = False
show_depth_image = True
save_depth_image = False
show_mask_image = True

#### GUI TODO Variables we will want to set in the GUI
## type_of_tracking =  'color' or 'infrared' or ...
    ## if type_of_tracking =='color': input_folder = 'color_i' ;  data_output = 'color_o'
    ## if type_of_tracking =='infrared': input_folder = 'color_i' ;  data_output = 'color_o'

## Default in GUI TODO Create a dialog box that explains what this is. 
## See user_input.py -> select_pixel_radius for text to use
min_radius_of_object = '5' 


# select the color *.npy files to use, folder to put the data in, and the clipping distance
# in the future it will also change the origin to zeroed__
## GUI TODO break up function and have each component get set in the GUI
file_type, data_output, color_ranges, min_radius_of_object, max_num_point = set_up_color(src)

## GUI TODO Have GUI real existing folders and provide an optional name for a new folder name being the last one listed with the number incremented by 1
data_output_folder, data_output_folder_path = make_new_folder(data_output)
## GUI TODO this needs to be done when new folder is created above
make_csv_files(color_ranges, data_output_folder_path)

# Configure and setup the cameras
pipeline = find_and_config_device()
# OpenCV initialization
cv_video = open_the_video(src) 

# Find the furthest distance and in the future find a different origin TODO
zeroed_x, zeroed_y, zeroed_z, clipping_distance = select_furthest_distance_color(cv_video, pipeline)

# Now that everything is setup, track the objects
first_time_check = True
start_time = 0 # It should get a time the first round through
i=0

while True:
    # Get frames if valid
    frame_result = get_all_frames_color(cv_video, pipeline)
    if not frame_result:
        continue
    
    (cv_color, rs_color, rs_depth), timestamp = frame_result
        
    ## Color Tracking by making a mask for each color tracked
    hsv = make_color_hsv(cv_color)
    for (lower,upper, color_name, radius_meters, mass) in color_ranges:
        # Find location of the object in x,y pixels using color masks
        x_pixel, y_pixel, radius, mask = find_object_by_color(cv_color,hsv, lower,upper, color_name, radius_meters, mass, min_radius_of_object, max_num_point)     

        if x_pixel is None:
            continue
        # get.distance is a little slower so only use if necessarycenter = round(aligned_depth_frame.get_distance(int(x),int(y)),4)

        x_coord, y_coord, z_coord = get_depth_meters(x_pixel, y_pixel, radius_meters, rs_depth, rs_color, zeroed_x, zeroed_y, zeroed_z, clipping_distance)
        if x_coord is None:
            continue
        # Append to the file until there is an error at which it will close
        
                # Start the timer
        if first_time_check:
            start_time = timestamp
            # we might want this later and compare with start that has milliseconds
            first_time_check = False

        # Converts time from milliseconds to seconds
        relative_timestamp = (timestamp - start_time) / 1000

        # Writes the coordinates to each colored object
        csv_file_path = os.path.abspath(os.path.join(data_output_folder_path, color_name + '.csv'))   
        with open(csv_file_path, 'a') as data_to_file:
            data_to_file.write(f'{relative_timestamp},{x_coord},{y_coord},{z_coord}\n')      
        
        cv2.putText(cv_color, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        cv2.putText(cv_color, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        cv2.putText(cv_color, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        cv2.putText(cv_color, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        
    depth_image = np.asanyarray(rs_depth.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)# Create a colormap from the depth data
    if show_depth_image:
        # Show depth colormap & color feed
        cv2.imshow('depth', depth_colormap)
        cv2.moveWindow('depth',700,0)

    if show_image:
        cv2.imshow('Tracking', cv_color)
        cv2.moveWindow('Tracking',0,0)
    
    if show_mask_image:
        cv2.imshow('mask', mask)
        cv2.moveWindow('mask',0,400)
    i +=1

    # Save the RGB and depth images to view later if you want, but it does slow the tracking down a bit.
    if save_image:
        color_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'color'+  str(i) + '.jpg'))   
        cv2.imwrite(color_file_path,cv_color)
    if save_depth_image:
        depth_file_path = os.path.abspath(os.path.join(data_output_folder_path, 'depth'+  str(i) + '.jpg'))   
        cv2.imwrite(depth_file_path, depth_colormap)
    # Show masks
    # ## cv2.imshow('mask', mask)
    ## cv2.moveWindow('mask',1000,0)
    
    # exit if spacebar or esc is pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 or k == 32:
        print('end')
        break

# Cleanup after loop break

# Stop OpenCV/RealSense video
cv_video.release()
pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()
