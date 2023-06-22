## Take the recorded bag file from IntelRealSense 2.0 software and track it with color for now.
## Settings for the framerate and screen resolution must match our program when recording.
## TODO make it be for any tracking?
## 

import time
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import pyrealsense2 as rs

from tracker.lib.setup_files import set_up_color, set_up_id, make_csv_files, make_csv_files_no_color

from tracker.lib.intel_realsense_D435i import get_all_frames_color, get_depth_meters, find_and_config_device, select_furthest_distance_color, read_bag_file_and_config, select_clipping_distance 
from tracker.lib.color import find_Time_x_y_z_from_rs_frames
from tracker.lib.infrared import find_object_by_subtracting_background, find_object_by_tracker
from tracker.lib.user_input import select_object_tracker_method, show_folders, select_files_to_graph, select_image_to_read_first, make_new_folder_trial


def output(i, save_image, type_of_frame, folder_path, frame, relative_timestamp, x_coord, y_coord, z_coord, color_ranges):
    cv2.putText(frame, 'Time: ' + str(relative_timestamp), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    #cv2.putText(frame, 'X coordinate: ' + str(x_coord), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    #cv2.putText(frame, 'Y coordinate: ' + str(y_coord), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    #cv2.putText(frame, 'Z coordinate: ' + str(z_coord), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

    cv2.imshow(type_of_frame, frame)
    if type_of_frame == 'color':
        location = 0
    else:
        location = 500
    
    #cv2.moveWindow(type_of_frame,location,0)
    
    cv2.imwrite(folder_path + '/' + type_of_frame + str(i).zfill(6) +'.jpg',frame)      

     

## Main Program ##
# TODO use argparse values in config file later
ap = argparse.ArgumentParser()

ap.add_argument("-s", "--src", default =0,
    help="video camera the port uses")

    
args = vars(ap.parse_args())
src = args["src"]

headerlist = ['Time', 'x', 'y', 'z']
# TODO Maybe add this later:
#ap.add_argument("-v", "--video",
#    help="path to the (optional) video file")

# select the color *.npy files to use, folder to put the data in, and the clipping distance
# in the future it will also change the origin to zeroed__

  
## TODO Test to see if they entered the the type available to the file selected
types_of_streams_saved = input('What streams do you want to read? \n (cd) color/depth 60 fps  \n (id) infrared/depth 90 fps \n (id300) infrared/depth at 300fps \n (all) color/infrared/depth 60 fps \n')

if 'all' in types_of_streams_saved:
    type_of_tracking = input('(c)olor  \n(deep) Learning \n(back)ground subtraction with infrared? \n')
elif 'cd' in types_of_streams_saved:
    type_of_tracking = input('(c)olor \n(deep) Learning \n ')
    data_folder = 'color_o'
elif 'infrared' in types_of_streams_saved:
    data_folder = 'infrared_o'
    type_of_tracking = input('(obj_tracker) track by shape \n (deep) Learning \n(backd)ground subtraction with depth \n(back)ground subtraction with infrared?')
elif 'id300' in types_of_streams_saved:
    data_folder = 'infrared_o'
    type_of_tracking = input('(tracker) track by shape \n (deep) Learning \n(backd)ground subtraction with depth \n(back)ground subtraction with infrared?')    

# What folders are currently in the output folder'
dir_path  = show_folders(data_folder)

# What folder is the bag file located?
while True:
    data_output_folder_path, data_output_folder = select_files_to_graph(dir_path)
    if os.path.exists(data_output_folder_path):
        break
    else:
        print('Name a folder that exists')

# Show the user the first image from the bag file you want to read.
jpg_image, first_image = select_image_to_read_first(data_output_folder_path, type_of_tracking)

# The image file is the cv_color when using color tracking
cv_color = jpg_image

# Find the furthest distance and in the future find a different origin TODO
#zeroed_x, zeroed_y, zeroed_z, clipping_distance = select_furthest_distance_color(cv_video, pipeline)

# Track the objects
first_time_check = True
start_time = 0 # It should get a time the first round through
i= -1

# What folder is the bag file located?
    
bag_file = str(data_output_folder + '.bag')
bag_folder_path =  os.path.abspath(os.path.join(data_output_folder_path + "/" + bag_file))    

# Read the bag file and get the pipeline of information
pipeline = read_bag_file_and_config(types_of_streams_saved, data_output_folder_path, data_output_folder , bag_folder_path)

# based on the folder you chose, choose the type of tracking you want to do with it
for i in range (first_image):
    rs_frames = pipeline.poll_for_frames()
    print('not using', i)
# Setup depends on type of tracking
 
#frame_result = Tracker_by_color.get_all_frames(pipeline)
# first image to be read is going to be shown to get clipping distance
        

# first image to be read is going to be shown to get clipping distance

 # When the readable image is no longer true, the *.bag file is done reading   
continue_reading = True
end_program = False
# search bag file for the first bright image that we can read
while continue_reading is True:
    i +=1 
    rs_frames = pipeline.wait_for_frames()
    if not rs_frames:
        print("couldn't read frame", i)
        if i>5000:
            pipeline.stop()
            end_program = True
            break

        continue
    else:
        break
if not end_program:

    if 'cd' in types_of_streams_saved:
        rs_depth = rs_frames.get_depth_frame().as_depth_frame()
        rs_color = rs_frames.get_color_frame()
        rs_align = rs.align(rs.stream.color)
        rs_frames_aligned = rs_align.process(rs_frames)

        # Extract color/depth frames
        rs_color = rs_frames_aligned.get_color_frame()
        rs_depth = rs_frames_aligned.get_depth_frame().as_depth_frame()
        cv_color = np.asanyarray(rs_color.get_data())

    elif 'id' in types_of_streams_saved:
    # If using the infrared at 90 fps or 300 fps
        rs_depth = rs_frames.get_depth_frame().as_depth_frame()
        rs_infrared = rs_frames.get_infrared_frame(1) 
        backg_depth = np.asanyarray(rs_depth.get_data())
        backg_infra = np.array(rs_infrared.get_data()) 
        if type_of_tracking =='backd':
            backgd = backg_depth
                
        else: backgd = backg_infra
                
        #color_ranges, min_radius_of_object, max_num_point, video= Color_masks.setup_color_tracking(color_image)
        
        #temporary code TODO get this to just display the color_image
        '''
        wall = cv2.selectROI('Select a clipping distance. Is will be the upper left of box)', cv_color, False)
        #cv2.imshow('check image', color_image)

        check_too_dark = input ('Is the image too dark? (y) or (n)')
        '''
        
    ## Determine the object to track and the location to put it.




    # Determine what object to track,how precise to track it, and setup a csv file for the data
    if 'cd' in types_of_streams_saved:
        file_type, data_output, object_ranges, min_radius_of_object, max_num_point = set_up_color(cv_color)
        # Make files to store data and put a header on the first row
        make_csv_files(object_ranges, data_output_folder_path)
        wall, depth, clipping_distance = select_clipping_distance(cv_color, rs_depth)
    elif 'infrared' in types_of_streams_saved:
        file_type, data_output, object_ranges, min_radius_of_object, max_num_point = set_up_id()
        # Make files to store data and put a header on the first row
        
        wall, depth, clipping_distance = select_clipping_distance(backg_infra, rs_depth)
        make_csv_files_no_color(object_ranges, data_output_folder_path)

        if type_of_tracking == 'obj_tracker':
            # User inputs the type of tracking used
            tracker = select_object_tracker_method()
            # Select the object to track
            print('select object')
            bbox = cv2.selectROI('ROI Selection', backg_infra, False)          
            ret = tracker.init(backg_infra, bbox)
            cv2.destroyWindow('ROI Selection')
    # Select the background distance, so it does not get tracked
    # Find the furthest distance and in the future find a different origin TODO

    # Don't select an object that is 95% of the distance to the wall behind




    # if type_of_tracking == 'back':
    show_image = True
    save_image = True
    first_time_check = True
    start_time = 0 # It should get a time the first round through
    save_depth_image = True
    timestamp = 0.0000
    zeroed_x, zeroed_y, zeroed_z = 0.0, 0.0, 0.0

while continue_reading is True:
    i += 1
    try:
        
        rs_frames = pipeline.wait_for_frames()
        timestamp = rs_frames.get_timestamp()
        print("Reading and frame:", i, timestamp)
        if not rs_frames:
            print("couldn't read", i)
            if i>6000:
                pipeline.stop()
                continue_reading = False
            continue
        
        # Start the timer
        if first_time_check:
            start_time = timestamp
            # we might want this later and compare with start that has miliseconds
            first_time_check = False
        # Converts time from miliseconds to seconds
        relative_timestamp = (timestamp - start_time) / 1000
        print (timestamp,'    ',relative_timestamp)
        
        if types_of_streams_saved =='cd': 
            # Color frames
            rs_align = rs.align(rs.stream.color)
            rs_frames_aligned = rs_align.process(rs_frames)

            # Extract color/depth frames
            rs_color = rs_frames_aligned.get_color_frame()
            rs_depth = rs_frames_aligned.get_depth_frame().as_depth_frame()
            # Create depth numpy array
            depth_image = np.asanyarray(rs_depth.get_data())
            # TODO erase if works 
            # rs_color = rs_frames.get_color_frame()
            cv_color = np.array(rs_color.get_data())
            #cv2.imwrite(folderPath + "/color"+ str(i).zfill(6) +  ".png", color_image)
            x_coord, y_coord, z_coord = find_Time_x_y_z_from_rs_frames(i, save_image, show_image, cv_color, rs_color, rs_depth, relative_timestamp, data_output_folder_path, object_ranges, zeroed_x, zeroed_y, zeroed_z, clipping_distance, min_radius_of_object, max_num_point)
            
            # Show the timestamp on the color image
            #output(i, save_image, 'color' , data_output_folder_path, cv_color, relative_timestamp, x_coord, y_coord, z_coord, object_ranges)


        elif 'id' in types_of_streams_saved: 
            rs_depth = rs_frames.get_depth_frame()
            depth_image = np.asanyarray(rs_depth.get_data())
            #infrared frames
            rs_infrared = rs_frames.get_infrared_frame(1)
            print ('got it', timestamp)

            #If using the depth instead of the monochrome color from infrared image
            if type_of_tracking =='backd':
                image = depth_image
  
            else:
                image = np.array(rs_infrared.get_data())
            if type_of_tracking == 'obj_tracker':
                ret, bbox = tracker.update(image)
                if ret:
                    print('found a frame')
                    find_object_by_tracker(bbox, i, relative_timestamp, backgd, image, rs_depth, rs_infrared, data_output_folder_path, type_of_tracking, zeroed_x, zeroed_y, zeroed_z, clipping_distance, object_ranges, max_num_point)
            else: 
                try:
                    x_coord, y_coord, z_coord = find_object_by_subtracting_background(i, relative_timestamp, backgd, image, rs_depth, rs_infrared, data_output_folder_path, type_of_tracking, zeroed_x, zeroed_y, zeroed_z, clipping_distance, object_ranges, min_radius_of_object, max_num_point)
                except Exception as e:
                    print ('Failed to get x_coord from background subtraction',e)
                    x_coord, y_coord, z_coord = None, None, None
        
            #cv2.imwrite(folderPath + "/infrared"+ str(i).zfill(6) +  ".png", infrared_image)

        
        

            
        # Show colormap of depth image and write it to a file
        #if save_depth_image == True:
        #    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), cv2.COLORMAP_HSV)
        #    output(i, save_image, 'depth' , data_output_folder_path, depth_colormap,relative_timestamp, x_coord, y_coord, z_coord, None)
        # Based on the type of tracking method selected, run different parts of other code.
        # if type_of_tracking == 'back':
        

        # exit if spacebar or esc is pressed
        k = cv2.waitKey(1) & 0xff
        if k ==27 or k==32:
            pipeline.stop()
            print ('end')
            break
    
    
    except RuntimeError:
        print("There are no more frames left in the .bag file!")
        pipeline.stop()
        continue_reading = False
        break
    finally:
        pass
    


#------------------------------------
  # https://stackoverflow.com/questions/58482414/frame-didnt-arrived-within-5000-while-reading-bag-file-pyrealsense2  


#        playback.resume()
#
#except RuntimeError:
#    print("There are no more frames left in the .bag file!")
#

